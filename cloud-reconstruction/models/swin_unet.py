"""models/swin_unet.py — Swin Transformer U-Net for Sentinel-2 cloud reconstruction.

Architecture:
    Encoder  : Swin Transformer Tiny (4 stages, shifted-window MSA)
    Decoder  : Symmetric Swin Transformer decoder with Patch Expanding
    Skip Conn: Concatenation + linear projection at each level

Input : [B, 6, H, W] optical  +  [B, 1, H, W] cloud mask  →  concatenated to [B, 7, H, W]
Output: [B, 6, H, W] reconstructed optical bands in [0, 1]

Minimum spatial_multiple = patch_size × window_size × 2^(num_stages-1)
  e.g. 4 × 8 × 8 = 256 for the default Swin-T config.
All inputs are reflect-padded to this multiple before encoding.

Gradient checkpointing is supported for VRAM reduction (~30-40%) at the
cost of extra FLOPs (recomputes activations during backward pass).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.utils.checkpoint as gradient_checkpoint
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Zero-pad spatial dims to the nearest multiple of `multiple`.

    Uses constant (zero) padding — no size constraint, unlike reflect mode.
    Returns (padded_tensor, (pad_h, pad_w)) so the padding can be removed after decoding.
    """
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return x, (pad_h, pad_w)


def _crop(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """Remove padding added by _pad_to_multiple."""
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition [B, H, W, C] into windows of shape [num_windows*B, window_size, window_size, C]."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window_partition. Returns [B, H, W, C]."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Window Attention (W-MSA / SW-MSA)
# ─────────────────────────────────────────────────────────────────────────────

class WindowAttention(nn.Module):
    """Window-based Multi-Head Self-Attention with relative position bias.

    Supports both regular (shift=0) and shifted (shift>0) windows.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # [2, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 2]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [N, N]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [num_windows*B, N, C] where N = window_size^2.
            mask: [num_windows, N, N] or None.
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B_, heads, N, N]

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [heads, N, N]
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply attention mask for shifted windows
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Swin Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with W-MSA or SW-MSA.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size for attention.
        shift_size: Shift size for SW-MSA (0 for W-MSA).
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = nn.Identity() if drop_path <= 0.0 else DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def _compute_attn_mask(self, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        """Compute attention mask for shifted windows."""
        if self.shift_size == 0:
            return None

        # Create image-level mask for shifted windows
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, ws, ws, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, C] flattened feature map.
            H, W: Spatial dimensions (needed for windowing).
        """
        B, L, C = x.shape
        assert L == H * W, f"Input length {L} != H*W ({H}*{W})"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, ws, ws, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_mask = self._compute_attn_mask(H, W, x.device)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor_(random_tensor + keep_prob)
        return x.div(keep_prob) * random_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Patch Embedding / Merging / Expanding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Image to Patch Embedding via Conv2d.

    Splits [B, C, H, W] → [B, H/patch_size * W/patch_size, embed_dim].
    """

    def __init__(self, in_chans: int = 7, embed_dim: int = 96, patch_size: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Returns (tokens [B, N, C], H_patches, W_patches)."""
        x = self.proj(x)  # [B, embed_dim, H/ps, W/ps]
        _, _, Hp, Wp = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        return x, Hp, Wp


class PatchMerging(nn.Module):
    """Downsample spatial resolution by 2× (merge 2×2 patches).

    Input : [B, H*W, C]
    Output: [B, H/2*W/2, 2C]
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)
        # Merge 2×2 patches
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4C]
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2C]
        return x, H // 2, W // 2


class PatchExpanding(nn.Module):
    """Upsample spatial resolution by 2× (inverse of PatchMerging).

    Input : [B, H*W, C]
    Output: [B, 4*H*W, C/2]  →  reshapes to [B, 2H*2W, C/2]
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        assert L == H * W

        x = self.expand(x)  # [B, H*W, 2C]
        x = x.view(B, H, W, 2 * C)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2)
        x = x.view(B, -1, C // 2)
        x = self.norm(x)
        return x, H * 2, W * 2


# ─────────────────────────────────────────────────────────────────────────────
# Encoder / Decoder Stage
# ─────────────────────────────────────────────────────────────────────────────

class SwinEncoderStage(nn.Module):
    """One encoder stage: N Swin Transformer blocks + optional PatchMerging.

    Args:
        use_checkpoint: Enable gradient checkpointing for this stage.
    """

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int,
                 downsample: bool = True, drop_path: list[float] | None = None,
                 use_checkpoint: bool = False) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                drop_path=drop_path[i] if drop_path else 0.0,
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Returns (downsampled_x, skip_x_before_downsample, new_H, new_W)."""
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                # Wrap block to accept (x, H, W) — checkpoint requires tensor args only.
                # H, W are python ints so we pass them as constant tensors.
                def _blk_fn(x, blk=blk, H=H, W=W):
                    return blk(x, H, W)
                x = gradient_checkpoint.checkpoint(_blk_fn, x, use_reentrant=False)
            else:
                x = blk(x, H, W)
        skip = x  # Skip connection before downsampling

        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)

        return x, skip, H, W


class SwinDecoderStage(nn.Module):
    """One decoder stage: PatchExpanding + skip fusion + N Swin Transformer blocks.

    Args:
        use_checkpoint: Enable gradient checkpointing for this stage.
    """

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int,
                 skip_dim: int, drop_path: list[float] | None = None,
                 use_checkpoint: bool = False) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.upsample = PatchExpanding(dim)
        # After upsampling dim becomes dim//2, concatenate with skip (skip_dim) → project
        self.skip_proj = nn.Linear(dim // 2 + skip_dim, dim // 2)
        self.skip_norm = nn.LayerNorm(dim // 2)

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim // 2,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                drop_path=drop_path[i] if drop_path else 0.0,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, skip: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            x: [B, H*W, C] from previous decoder stage / bottleneck.
            skip: [B, 2H*2W, C/2] from corresponding encoder stage.
            H, W: Spatial dims of x BEFORE upsampling.
        """
        x, H, W = self.upsample(x, H, W)  # [B, 4*H*W, C/2] → reshapes

        # Fuse with skip connection
        x = torch.cat([x, skip], dim=-1)  # [B, H*W, C/2 + skip_dim]
        x = self.skip_proj(x)
        x = self.skip_norm(x)

        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                def _blk_fn(x, blk=blk, H=H, W=W):
                    return blk(x, H, W)
                x = gradient_checkpoint.checkpoint(_blk_fn, x, use_reentrant=False)
            else:
                x = blk(x, H, W)

        return x, H, W


# ─────────────────────────────────────────────────────────────────────────────
# Swin U-Net
# ─────────────────────────────────────────────────────────────────────────────

class SwinUNet(nn.Module):
    """
    Swin Transformer U-Net for cloud reconstruction.

    Input:
        cloudy    : [B, 6, H, W]  — 6 Sentinel-2 optical bands
        cloud_mask: [B, 1, H, W]  — binary cloud mask
        (Concatenated internally → [B, 7, H, W])

    Output:
        prediction: [B, 6, H, W]  — reconstructed bands in [0, 1]

    Architecture:
        PatchEmbed(7 → C) → Encoder(4 stages) → Bottleneck → Decoder(4 stages) → Head(C → 6)
        Skip connections between matching encoder/decoder stages.
        Shifted Window Multi-Head Self-Attention (W-MSA / SW-MSA) at every block.

    Parameters are read from cfg["model"]:
        embed_dim   : Base embedding dimension (default 96 = Swin-T).
        depths      : Number of blocks per stage (default [2, 2, 6, 2] = Swin-T).
        num_heads   : Attention heads per stage (default [3, 6, 12, 24] = Swin-T).
        window_size : Window size for attention (default 8).
        patch_size  : Patch embedding kernel size (default 4).
        pretrained  : Load ImageNet Swin-T weights for encoder (default True).
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        mcfg = cfg["model"]

        in_ch = mcfg.get("input_channels", 7)     # 6 optical + 1 mask
        out_ch = mcfg.get("output_channels", 6)
        embed_dim = mcfg.get("embed_dim", 96)
        depths = mcfg.get("depths", [2, 2, 6, 2])
        num_heads = mcfg.get("num_heads", [3, 6, 12, 24])
        window_size = mcfg.get("window_size", 8)
        patch_size = mcfg.get("patch_size", 4)
        use_pretrained = mcfg.get("pretrained", True)
        drop_path_rate = mcfg.get("drop_path_rate", 0.1)

        use_gradient_checkpointing = mcfg.get("gradient_checkpointing", True)

        self.patch_size = patch_size
        self.window_size = window_size
        # CRITICAL: minimum pad must ensure the deepest feature map (after num_stages-1
        # PatchMerging ops, each halving spatial dims) is still divisible by window_size.
        # spatial_multiple = patch_size * window_size * 2^(num_stages-1)
        # For default Swin-T (ps=4, ws=8, 4 stages): 4 * 8 * 8 = 256
        n_downsamplings = len(depths) - 1  # last stage has no downsample
        self.spatial_multiple = patch_size * window_size * (2 ** n_downsamplings)
        self.num_stages = len(depths)
        self.embed_dim = embed_dim

        # Stochastic depth decay
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # ── Patch Embedding ──
        self.patch_embed = PatchEmbed(in_chans=in_ch, embed_dim=embed_dim, patch_size=patch_size)

        # ── Encoder stages ──
        self.encoder_stages = nn.ModuleList()
        dims = []
        block_idx = 0
        for i in range(self.num_stages):
            stage_dim = embed_dim * (2 ** i)
            dims.append(stage_dim)
            stage_depth = depths[i]
            self.encoder_stages.append(SwinEncoderStage(
                dim=stage_dim,
                depth=stage_depth,
                num_heads=num_heads[i],
                window_size=window_size,
                downsample=(i < self.num_stages - 1),
                drop_path=dpr[block_idx:block_idx + stage_depth],
                use_checkpoint=use_gradient_checkpointing,
            ))
            block_idx += stage_depth

        # ── Bottleneck ──
        bottleneck_dim = dims[-1]  # Same as last encoder stage (no extra downsample)
        self.bottleneck_norm = nn.LayerNorm(bottleneck_dim)

        # ── Decoder stages ──
        self.decoder_stages = nn.ModuleList()
        # Decoder goes from deepest to shallowest
        # Stage i decoder: input dim = dims[num_stages-1-i] after merging (if applicable)
        # For a 4-stage encoder with dims [C, 2C, 4C, 8C]:
        #   Last encoder stage (no downsample) outputs 8C
        #   Decoder stages receive: 8C→4C, 4C→2C, 2C→C
        #   But we only need num_stages-1 decoder stages since last encoder has no downsample
        for i in range(self.num_stages - 1):
            # Going from deep to shallow: indices are num_stages-1, num_stages-2, ...
            dec_input_dim = dims[self.num_stages - 1 - i]
            # IMPORTANT: skip connections come from encoder stages 0..num_stages-2 only.
            # Encoder stage num_stages-1 is the bottleneck input, NOT a skip.
            # Decoder stage 0 fuses with encoder stage num_stages-2 skip.
            # Decoder stage 1 fuses with encoder stage num_stages-3 skip.
            # Decoder stage i fuses with encoder stage num_stages-2-i skip.
            skip_enc_idx = self.num_stages - 2 - i   # always in [0, num_stages-2]
            skip_dim = dims[skip_enc_idx]
            dec_heads = num_heads[skip_enc_idx]
            dec_depth = depths[skip_enc_idx]

            self.decoder_stages.append(SwinDecoderStage(
                dim=dec_input_dim,
                depth=dec_depth,
                num_heads=dec_heads,
                window_size=window_size,
                skip_dim=skip_dim,
                use_checkpoint=use_gradient_checkpointing,
            ))

        # ── Output head ──
        # After all decoder stages, feature tokens are at:
        #   shape [B, (H/ps)*(W/ps), embed_dim]  spatial resolution = H/ps x W/ps
        # We need to project each token to ps*ps*out_ch values and fold into [B, out_ch, H, W].
        #
        # We use a 2-layer MLP (embed_dim → 4*embed_dim → ps*ps*out_ch) for expressiveness.
        # The final bias is initialised to logit(0.35) so that sigmoid(output) starts near
        # the Sentinel-2 clear-sky mean (~0.35), giving large gradients from epoch 1.
        _expand_dim = 4 * embed_dim   # 384 for Swin-T
        self.final_expand = nn.Sequential(
            nn.Linear(embed_dim, _expand_dim, bias=True),
            nn.GELU(),
            nn.Linear(_expand_dim, patch_size * patch_size * out_ch, bias=True),
        )
        self.final_norm = nn.LayerNorm(patch_size * patch_size * out_ch)
        self.out_ch = out_ch

        # ── Initialize weights ──
        self.apply(self._init_weights)

        # Bias-init the final linear so sigmoid(output) ≈ 0.35 (Sentinel-2 clear mean)
        # logit(0.35) = log(0.35/0.65) ≈ -0.619
        # This centers predictions in the target distribution from epoch 1,
        # keeping sigmoid gradients large where the data actually lives.
        import math as _math
        _target_mean = 0.35
        _logit_bias  = _math.log(_target_mean / (1.0 - _target_mean))  # ≈ -0.619
        with torch.no_grad():
            self.final_expand[-1].bias.fill_(_logit_bias)

        # ── Load pretrained Swin-T encoder weights ──
        if use_pretrained:
            self._load_pretrained_encoder(in_ch)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights following Swin Transformer conventions."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _load_pretrained_encoder(self, in_ch: int) -> None:
        """Load pretrained Swin-T weights from timm into the encoder.

        Strategy:
        - Load the full pretrained Swin-T model.
        - Copy patch_embed weights (adapting for in_ch != 3 by repeating/truncating).
        - Copy each encoder stage's transformer blocks.
        - Copy patch merging layers.
        - Ignore classifier head.
        """
        try:
            import timm
            pretrained_model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
        except Exception as e:
            print(f"[SwinUNet] Could not load pretrained weights: {e}. Training from scratch.")
            return

        # ── Patch Embed ──
        pretrained_proj = pretrained_model.patch_embed.proj.weight.data  # [96, 3, 4, 4]
        my_proj = self.patch_embed.proj.weight.data  # [96, in_ch, 4, 4]

        if in_ch == 3:
            my_proj.copy_(pretrained_proj)
        elif in_ch > 3:
            # Copy RGB weights, then repeat for extra channels
            my_proj[:, :3] = pretrained_proj
            repeat_idx = 0
            for c in range(3, in_ch):
                my_proj[:, c] = pretrained_proj[:, repeat_idx % 3] * 0.1  # Scaled copy
                repeat_idx += 1
        else:
            my_proj.copy_(pretrained_proj[:, :in_ch])

        if self.patch_embed.proj.bias is not None and pretrained_model.patch_embed.proj.bias is not None:
            self.patch_embed.proj.bias.data.copy_(pretrained_model.patch_embed.proj.bias.data)

        # ── Patch Embed Norm ──
        if hasattr(pretrained_model.patch_embed, "norm") and pretrained_model.patch_embed.norm is not None:
            self.patch_embed.norm.weight.data.copy_(pretrained_model.patch_embed.norm.weight.data)
            self.patch_embed.norm.bias.data.copy_(pretrained_model.patch_embed.norm.bias.data)

        # ── Encoder Stages ──
        for stage_idx in range(min(self.num_stages, len(pretrained_model.layers))):
            pretrained_layer = pretrained_model.layers[stage_idx]
            my_stage = self.encoder_stages[stage_idx]

            # Copy transformer blocks
            for blk_idx in range(min(len(my_stage.blocks), len(pretrained_layer.blocks))):
                my_blk = my_stage.blocks[blk_idx]
                pre_blk = pretrained_layer.blocks[blk_idx]

                # Copy all matching parameters
                my_blk_sd = my_blk.state_dict()
                pre_blk_sd = pre_blk.state_dict()
                matched = {}
                for k in my_blk_sd:
                    # Map our naming to timm naming
                    if k in pre_blk_sd and my_blk_sd[k].shape == pre_blk_sd[k].shape:
                        matched[k] = pre_blk_sd[k]

                if matched:
                    my_blk.load_state_dict(matched, strict=False)

            # Copy downsample / patch merging if present
            if my_stage.downsample is not None and hasattr(pretrained_layer, "downsample") and pretrained_layer.downsample is not None:
                my_ds = my_stage.downsample
                pre_ds = pretrained_layer.downsample
                my_ds_sd = my_ds.state_dict()
                pre_ds_sd = pre_ds.state_dict()
                matched = {}
                for k in my_ds_sd:
                    if k in pre_ds_sd and my_ds_sd[k].shape == pre_ds_sd[k].shape:
                        matched[k] = pre_ds_sd[k]
                if matched:
                    my_ds.load_state_dict(matched, strict=False)

        del pretrained_model
        print("[SwinUNet] Loaded pretrained Swin-T encoder weights (ImageNet-1K).")

    def forward(
        self,
        cloudy: torch.Tensor,
        cloud_mask: torch.Tensor,
        sentinel1: Optional[torch.Tensor] = None,  # Unused, kept for API compat
    ) -> torch.Tensor:
        """
        Args:
            cloudy:     [B, 6, H, W] — 6 optical bands in [0, 1].
            cloud_mask: [B, 1, H, W] — binary cloud mask.

        Returns:
            prediction: [B, 6, H, W] — reconstructed bands in [0, 1].
        """
        # Concatenate optical + mask → [B, 7, H, W]
        if cloud_mask.dim() == 3:
            cloud_mask = cloud_mask.unsqueeze(1)
        x = torch.cat([cloudy, cloud_mask], dim=1)

        _, _, orig_H, orig_W = x.shape

        # Pad to multiple of spatial_multiple (32)
        x, (pad_h, pad_w) = _pad_to_multiple(x, self.spatial_multiple)
        _, _, H, W = x.shape

        # ── Patch Embedding ──
        x, Hp, Wp = self.patch_embed(x)  # [B, Hp*Wp, C]

        # ── Encoder ──
        skips: list[torch.Tensor] = []
        skip_dims: list[tuple[int, int]] = []  # (H, W) for each skip
        cur_H, cur_W = Hp, Wp

        for stage in self.encoder_stages:
            x, skip, cur_H, cur_W = stage(x, cur_H, cur_W)
            skips.append(skip)
            # skip has the spatial dims BEFORE downsampling
            if stage.downsample is not None:
                skip_dims.append((cur_H * 2, cur_W * 2))
            else:
                skip_dims.append((cur_H, cur_W))

        # ── Bottleneck ──
        x = self.bottleneck_norm(x)

        # ── Decoder ──
        # skips: [stage0_skip, stage1_skip, stage2_skip, stage3_skip]
        # decoder needs: stage2_skip, stage1_skip, stage0_skip (reversed, excluding last)
        for i, dec_stage in enumerate(self.decoder_stages):
            # Skip connections: encoder stage num_stages-2-i
            # (never uses the last encoder stage — that feeds the bottleneck)
            skip_enc_idx = self.num_stages - 2 - i
            skip = skips[skip_enc_idx]
            x, cur_H, cur_W = dec_stage(x, skip, cur_H, cur_W)

        # ── Output Head ──
        # x: [B, Hp*Wp, embed_dim]
        x = self.final_expand(x)  # [B, Hp*Wp, ps*ps*out_ch]
        x = self.final_norm(x)

        # Reshape to image
        x = x.view(-1, cur_H, cur_W, self.patch_size, self.patch_size, self.out_ch)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(-1, self.out_ch, cur_H * self.patch_size, cur_W * self.patch_size)

        # Sigmoid → [0, 1]
        # Bias is pre-initialised to logit(0.35) so predictions start near
        # the Sentinel-2 clear-sky mean, giving large sigmoid gradients at the
        # values the model actually needs to predict.
        x = torch.sigmoid(x)

        # Crop padding
        x = x[:, :, :orig_H, :orig_W]

        return x

    def get_config(self) -> dict:
        """Return model configuration for logging."""
        mcfg = self.cfg["model"]
        return {
            "model": "SwinUNet",
            "input_channels": mcfg.get("input_channels", 7),
            "output_channels": mcfg.get("output_channels", 6),
            "embed_dim": mcfg.get("embed_dim", 96),
            "depths": mcfg.get("depths", [2, 2, 6, 2]),
            "num_heads": mcfg.get("num_heads", [3, 6, 12, 24]),
            "window_size": mcfg.get("window_size", 8),
            "patch_size": mcfg.get("patch_size", 4),
            "parameters": self.count_parameters(),
        }

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
