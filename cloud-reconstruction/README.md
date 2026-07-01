# Cloud Reconstruction — AI-Based Cloud Removal & Surface Reconstruction

Production-grade deep learning system for reconstructing cloud-covered regions in Sentinel-2 satellite imagery while preserving real Earth features (roads, rivers, vegetation, buildings, water bodies, terrain).

## Architecture

```
Upload Image → Validation → GeoTIFF Reader → Band Extraction (B02/B03/B04/B08/B11/B12)
→ Cloud Detection (SCL + fallback) → Cloud Mask → Patch Extraction (256×256)
→ AI Reconstruction (U-Net) → Merge Patches (Gaussian blend)
→ Post Processing (blend cloud regions only) → Quality Assessment
→ Cloud-Free GeoTIFF + RGB Preview + Confidence Map
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare dataset (extract patches from .SAFE.zip)
```bash
python prepare_dataset.py
```

### 3. Train the model
```bash
# Full training (GPU recommended)
python train.py

# Custom settings
python train.py --epochs 50 --batch 4 --lr 0.0001

# Resume from checkpoint
python train.py --resume
```

### 4. Evaluate
```bash
python validate.py --split validation --save-panels
python validate.py --split test
```

### 5. Run inference
```bash
# Single image
python inference.py --input path/to/cloudy.tif

# .SAFE.zip
python inference.py --input path/to/scene.SAFE.zip

# Batch mode
python inference.py --batch-dir path/to/folder/
```

## Configuration

All settings in `config.yaml`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch.size` | 256 | Patch dimensions |
| `patch.stride` | 128 | 50% overlap |
| `bands.optical` | B02-B12 | 6 spectral bands |
| `model.name` | unet | Model architecture |
| `training.epochs` | 100 | Training epochs |
| `training.amp` | true | Mixed precision |
| `loss.l1/ssim/perceptual/edge` | 0.4/0.3/0.2/0.1 | Loss weights |

## Model Architecture

**U-Net** with:
- 4-level encoder-decoder with skip connections
- 7 input channels (6 optical bands + cloud mask)
- 6 output channels (reconstructed bands)
- FusionGate for Sentinel-1 SAR (extension point)
- Sigmoid output head

**Loss Function:**
```
L = 0.4×L1 + 0.3×SSIM + 0.2×VGG_Perceptual + 0.1×Sobel_Edge
```
Applied **only on cloud-masked pixels**.

## Outputs

Each inference run produces:
- `cloud_free.tif` — Multi-band GeoTIFF (CRS/transform preserved)
- `cloud_free_rgb.jpg` — RGB preview for web display
- `cloud_mask.png` — Binary cloud mask
- `confidence_map.png` — Per-pixel reconstruction confidence
- `difference_map.png` — Prediction vs input difference
- `comparison.jpg` — 5-panel side-by-side
- `metrics.json` — PSNR, SSIM, RMSE, processing time
- `processing_log.txt` — Full processing log

## Future Integration

The architecture supports plug-and-play model swapping:

```yaml
model:
  name: unet       # → pix2pix | diffusion | vit
```

Sentinel-1 SAR fusion:
```yaml
sentinel1:
  enabled: true    # activates FusionGate
  channels: 2      # VV + VH
```

## Project Structure

```
cloud-reconstruction/
  config.yaml               ← all settings
  prepare_dataset.py        ← Step 1
  train.py                  ← Step 2
  validate.py               ← Step 3
  inference.py              ← Step 4
  models/
    base_model.py           ← abstract interface
    unet.py                 ← main architecture
    losses.py               ← L1+SSIM+VGG+Edge
    registry.py             ← model factory
  preprocessing/
    extractor.py            ← .SAFE.zip band reader
    patcher.py              ← patch extract + Gaussian merge
    loader.py               ← PyTorch Dataset
    transforms.py           ← Albumentations augmentation
    validator.py            ← image validation
    cloud_detector.py       ← SCL + threshold detector
  training/
    trainer.py              ← AMP + early stopping + TensorBoard
  evaluation/
    metrics.py              ← PSNR/SSIM/RMSE
    confidence.py           ← per-pixel confidence
    visualizer.py           ← 5-panel comparisons
  utils/
    logger.py               ← TensorBoard + file logging
    checkpoint.py           ← save/load checkpoints
    geotiff.py              ← metadata-preserving I/O
```
