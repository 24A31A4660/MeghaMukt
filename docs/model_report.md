# Model Checkpoint Compatibility Report
**Checkpoint:** `c:\Users\aksha\Documents\AntiGravity\earth-satellites\cloud-reconstruction\checkpoints_6band\best.pth`
**Epoch:** `124`

## Layer Verification
| Layer Name | Model Shape | Checkpoint Shape | Status |
|------------|-------------|------------------|--------|
| `encoders.0.conv.block.0.weight` | `[64, 3, 3, 3]` | `[64, 7, 3, 3]` | ❌ MISMATCH |
| `encoders.0.conv.block.1.weight` | `[64]` | `[64]` | ✅ OK |
| `encoders.0.conv.block.1.bias` | `[64]` | `[64]` | ✅ OK |
| `encoders.0.conv.block.1.running_mean` | `[64]` | `[64]` | ✅ OK |
| `encoders.0.conv.block.1.running_var` | `[64]` | `[64]` | ✅ OK |
| `encoders.0.conv.block.1.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `encoders.0.conv.block.3.weight` | `[64, 64, 3, 3]` | `[64, 64, 3, 3]` | ✅ OK |
| `encoders.0.conv.block.4.weight` | `[64]` | `[64]` | ✅ OK |
| `encoders.0.conv.block.4.bias` | `[64]` | `[64]` | ✅ OK |
| `encoders.0.conv.block.4.running_mean` | `[64]` | `[64]` | ✅ OK |
| `encoders.0.conv.block.4.running_var` | `[64]` | `[64]` | ✅ OK |
| `encoders.0.conv.block.4.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `encoders.1.conv.block.0.weight` | `[128, 64, 3, 3]` | `[128, 64, 3, 3]` | ✅ OK |
| `encoders.1.conv.block.1.weight` | `[128]` | `[128]` | ✅ OK |
| `encoders.1.conv.block.1.bias` | `[128]` | `[128]` | ✅ OK |
| `encoders.1.conv.block.1.running_mean` | `[128]` | `[128]` | ✅ OK |
| `encoders.1.conv.block.1.running_var` | `[128]` | `[128]` | ✅ OK |
| `encoders.1.conv.block.1.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `encoders.1.conv.block.3.weight` | `[128, 128, 3, 3]` | `[128, 128, 3, 3]` | ✅ OK |
| `encoders.1.conv.block.4.weight` | `[128]` | `[128]` | ✅ OK |
| `encoders.1.conv.block.4.bias` | `[128]` | `[128]` | ✅ OK |
| `encoders.1.conv.block.4.running_mean` | `[128]` | `[128]` | ✅ OK |
| `encoders.1.conv.block.4.running_var` | `[128]` | `[128]` | ✅ OK |
| `encoders.1.conv.block.4.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `encoders.2.conv.block.0.weight` | `[256, 128, 3, 3]` | `[256, 128, 3, 3]` | ✅ OK |
| `encoders.2.conv.block.1.weight` | `[256]` | `[256]` | ✅ OK |
| `encoders.2.conv.block.1.bias` | `[256]` | `[256]` | ✅ OK |
| `encoders.2.conv.block.1.running_mean` | `[256]` | `[256]` | ✅ OK |
| `encoders.2.conv.block.1.running_var` | `[256]` | `[256]` | ✅ OK |
| `encoders.2.conv.block.1.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `encoders.2.conv.block.3.weight` | `[256, 256, 3, 3]` | `[256, 256, 3, 3]` | ✅ OK |
| `encoders.2.conv.block.4.weight` | `[256]` | `[256]` | ✅ OK |
| `encoders.2.conv.block.4.bias` | `[256]` | `[256]` | ✅ OK |
| `encoders.2.conv.block.4.running_mean` | `[256]` | `[256]` | ✅ OK |
| `encoders.2.conv.block.4.running_var` | `[256]` | `[256]` | ✅ OK |
| `encoders.2.conv.block.4.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `encoders.3.conv.block.0.weight` | `[512, 256, 3, 3]` | `[512, 256, 3, 3]` | ✅ OK |
| `encoders.3.conv.block.1.weight` | `[512]` | `[512]` | ✅ OK |
| `encoders.3.conv.block.1.bias` | `[512]` | `[512]` | ✅ OK |
| `encoders.3.conv.block.1.running_mean` | `[512]` | `[512]` | ✅ OK |
| `encoders.3.conv.block.1.running_var` | `[512]` | `[512]` | ✅ OK |
| `encoders.3.conv.block.1.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `encoders.3.conv.block.3.weight` | `[512, 512, 3, 3]` | `[512, 512, 3, 3]` | ✅ OK |
| `encoders.3.conv.block.4.weight` | `[512]` | `[512]` | ✅ OK |
| `encoders.3.conv.block.4.bias` | `[512]` | `[512]` | ✅ OK |
| `encoders.3.conv.block.4.running_mean` | `[512]` | `[512]` | ✅ OK |
| `encoders.3.conv.block.4.running_var` | `[512]` | `[512]` | ✅ OK |
| `encoders.3.conv.block.4.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `bottleneck.block.0.weight` | `[1024, 512, 3, 3]` | `[1024, 512, 3, 3]` | ✅ OK |
| `bottleneck.block.1.weight` | `[1024]` | `[1024]` | ✅ OK |
| `bottleneck.block.1.bias` | `[1024]` | `[1024]` | ✅ OK |
| `bottleneck.block.1.running_mean` | `[1024]` | `[1024]` | ✅ OK |
| `bottleneck.block.1.running_var` | `[1024]` | `[1024]` | ✅ OK |
| `bottleneck.block.1.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `bottleneck.block.3.weight` | `[1024, 1024, 3, 3]` | `[1024, 1024, 3, 3]` | ✅ OK |
| `bottleneck.block.4.weight` | `[1024]` | `[1024]` | ✅ OK |
| `bottleneck.block.4.bias` | `[1024]` | `[1024]` | ✅ OK |
| `bottleneck.block.4.running_mean` | `[1024]` | `[1024]` | ✅ OK |
| `bottleneck.block.4.running_var` | `[1024]` | `[1024]` | ✅ OK |
| `bottleneck.block.4.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `decoders.0.up.weight` | `[1024, 512, 2, 2]` | `[1024, 512, 2, 2]` | ✅ OK |
| `decoders.0.up.bias` | `[512]` | `[512]` | ✅ OK |
| `decoders.0.conv.block.0.weight` | `[512, 1024, 3, 3]` | `[512, 1024, 3, 3]` | ✅ OK |
| `decoders.0.conv.block.1.weight` | `[512]` | `[512]` | ✅ OK |
| `decoders.0.conv.block.1.bias` | `[512]` | `[512]` | ✅ OK |
| `decoders.0.conv.block.1.running_mean` | `[512]` | `[512]` | ✅ OK |
| `decoders.0.conv.block.1.running_var` | `[512]` | `[512]` | ✅ OK |
| `decoders.0.conv.block.1.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `decoders.0.conv.block.3.weight` | `[512, 512, 3, 3]` | `[512, 512, 3, 3]` | ✅ OK |
| `decoders.0.conv.block.4.weight` | `[512]` | `[512]` | ✅ OK |
| `decoders.0.conv.block.4.bias` | `[512]` | `[512]` | ✅ OK |
| `decoders.0.conv.block.4.running_mean` | `[512]` | `[512]` | ✅ OK |
| `decoders.0.conv.block.4.running_var` | `[512]` | `[512]` | ✅ OK |
| `decoders.0.conv.block.4.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `decoders.1.up.weight` | `[512, 256, 2, 2]` | `[512, 256, 2, 2]` | ✅ OK |
| `decoders.1.up.bias` | `[256]` | `[256]` | ✅ OK |
| `decoders.1.conv.block.0.weight` | `[256, 512, 3, 3]` | `[256, 512, 3, 3]` | ✅ OK |
| `decoders.1.conv.block.1.weight` | `[256]` | `[256]` | ✅ OK |
| `decoders.1.conv.block.1.bias` | `[256]` | `[256]` | ✅ OK |
| `decoders.1.conv.block.1.running_mean` | `[256]` | `[256]` | ✅ OK |
| `decoders.1.conv.block.1.running_var` | `[256]` | `[256]` | ✅ OK |
| `decoders.1.conv.block.1.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `decoders.1.conv.block.3.weight` | `[256, 256, 3, 3]` | `[256, 256, 3, 3]` | ✅ OK |
| `decoders.1.conv.block.4.weight` | `[256]` | `[256]` | ✅ OK |
| `decoders.1.conv.block.4.bias` | `[256]` | `[256]` | ✅ OK |
| `decoders.1.conv.block.4.running_mean` | `[256]` | `[256]` | ✅ OK |
| `decoders.1.conv.block.4.running_var` | `[256]` | `[256]` | ✅ OK |
| `decoders.1.conv.block.4.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `decoders.2.up.weight` | `[256, 128, 2, 2]` | `[256, 128, 2, 2]` | ✅ OK |
| `decoders.2.up.bias` | `[128]` | `[128]` | ✅ OK |
| `decoders.2.conv.block.0.weight` | `[128, 256, 3, 3]` | `[128, 256, 3, 3]` | ✅ OK |
| `decoders.2.conv.block.1.weight` | `[128]` | `[128]` | ✅ OK |
| `decoders.2.conv.block.1.bias` | `[128]` | `[128]` | ✅ OK |
| `decoders.2.conv.block.1.running_mean` | `[128]` | `[128]` | ✅ OK |
| `decoders.2.conv.block.1.running_var` | `[128]` | `[128]` | ✅ OK |
| `decoders.2.conv.block.1.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `decoders.2.conv.block.3.weight` | `[128, 128, 3, 3]` | `[128, 128, 3, 3]` | ✅ OK |
| `decoders.2.conv.block.4.weight` | `[128]` | `[128]` | ✅ OK |
| `decoders.2.conv.block.4.bias` | `[128]` | `[128]` | ✅ OK |
| `decoders.2.conv.block.4.running_mean` | `[128]` | `[128]` | ✅ OK |
| `decoders.2.conv.block.4.running_var` | `[128]` | `[128]` | ✅ OK |
| `decoders.2.conv.block.4.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `decoders.3.up.weight` | `[128, 64, 2, 2]` | `[128, 64, 2, 2]` | ✅ OK |
| `decoders.3.up.bias` | `[64]` | `[64]` | ✅ OK |
| `decoders.3.conv.block.0.weight` | `[64, 128, 3, 3]` | `[64, 128, 3, 3]` | ✅ OK |
| `decoders.3.conv.block.1.weight` | `[64]` | `[64]` | ✅ OK |
| `decoders.3.conv.block.1.bias` | `[64]` | `[64]` | ✅ OK |
| `decoders.3.conv.block.1.running_mean` | `[64]` | `[64]` | ✅ OK |
| `decoders.3.conv.block.1.running_var` | `[64]` | `[64]` | ✅ OK |
| `decoders.3.conv.block.1.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `decoders.3.conv.block.3.weight` | `[64, 64, 3, 3]` | `[64, 64, 3, 3]` | ✅ OK |
| `decoders.3.conv.block.4.weight` | `[64]` | `[64]` | ✅ OK |
| `decoders.3.conv.block.4.bias` | `[64]` | `[64]` | ✅ OK |
| `decoders.3.conv.block.4.running_mean` | `[64]` | `[64]` | ✅ OK |
| `decoders.3.conv.block.4.running_var` | `[64]` | `[64]` | ✅ OK |
| `decoders.3.conv.block.4.num_batches_tracked` | `[]` | `[]` | ✅ OK |
| `head.0.weight` | `[3, 64, 1, 1]` | `[6, 64, 1, 1]` | ❌ MISMATCH |
| `head.0.bias` | `[3]` | `[6]` | ❌ MISMATCH |

## Summary
❌ **Mismatches Detected!**

### Shape Mismatches
- **encoders.0.conv.block.0.weight**: Model expects `[64, 3, 3, 3]`, but checkpoint has `[64, 7, 3, 3]`
- **head.0.weight**: Model expects `[3, 64, 1, 1]`, but checkpoint has `[6, 64, 1, 1]`
- **head.0.bias**: Model expects `[3]`, but checkpoint has `[6]`

## Configuration Architecture
| Parameter | Current Config | Checkpoint Config |
|-----------|----------------|-------------------|
| input_channels | 3 | 7 | ❌ |
| dropout | 0.1 | 0.1 | ✅ |
| output_channels | 3 | 6 | ❌ |
| depth | 4 | 4 | ✅ |
| base_filters | 64 | 64 | ✅ |
| name | unet | unet | ✅ |
| use_batchnorm | True | True | ✅ |