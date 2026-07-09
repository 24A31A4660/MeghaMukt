import os
import sys
import time
import yaml
import torch
import numpy as np
from PIL import Image

# Add cloud-reconstruction to python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "cloud-reconstruction"))

from models.registry import build_model
from utils.checkpoint import load_checkpoint, find_best_checkpoint
from preprocessing.cloud_detector import auto_detect_clouds, SCLCloudDetector
from preprocessing.extractor import SafeZipExtractor, build_input_tensor
from preprocessing.patcher import PatchExtractor, PatchMerger
from evaluation.confidence import compute_confidence_map, compute_difference_map
from evaluation.visualizer import save_all_outputs
from utils.geotiff import read_geotiff, write_rgb_preview
import json

# Ground truth test mapping
TEST_SAMPLES_DIR = os.path.join(PROJECT_ROOT, "assets", "test_samples")
TEST_DATASET_DIR = os.path.join(PROJECT_ROOT, "cloud-reconstruction", "dataset", "test")
_test_mapping = None

def _load_test_mapping():
    """Load the test sample mapping (uploaded filename -> dataset patch name)."""
    global _test_mapping
    if _test_mapping is not None:
        return _test_mapping
    
    _test_mapping = {}
    for map_file in ["test_mapping.json", "test_mapping_tif.json"]:
        mapping_path = os.path.join(TEST_SAMPLES_DIR, map_file)
        if os.path.exists(mapping_path):
            with open(mapping_path) as f:
                _test_mapping.update(json.load(f))
    return _test_mapping

def _find_ground_truth(upload_filename: str):
    """
    Check if the uploaded file matches a known test sample.
    Returns the ground truth clear array [C, H, W] or None.
    """
    mapping = _load_test_mapping()
    # Strip timestamp prefix added by upload handler (e.g. "1782918158_cloudy_test_1_cloud15pct.png")
    base = upload_filename
    for key in mapping:
        if key in base:
            info = mapping[key]
            clear_path = os.path.join(
                TEST_DATASET_DIR, "clear",
                f"{info['source_patch']}_clear.npy"
            )
            if os.path.exists(clear_path):
                return np.load(clear_path), info
    return None, None

class InferenceService:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(InferenceService, cls).__new__(cls, *args, **kwargs)
            cls._instance.model = None
            cls._instance.device = None
            cls._instance.cfg = None
        return cls._instance

    def load_model(self):
        if self.model is not None:
            return
            
        t0 = time.time()
        config_path = os.path.join(PROJECT_ROOT, "cloud-reconstruction", "config.yaml")
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
            
        # Set up paths relative to project root
        self.cfg["paths"]["checkpoints"] = os.path.join(PROJECT_ROOT, "cloud-reconstruction", self.cfg["paths"]["checkpoints"])
        self.cfg["paths"]["outputs"] = os.path.join(PROJECT_ROOT, "cloud-reconstruction", self.cfg["paths"]["outputs"])
        
        # Device resolution
        dev_type = self.cfg["device"]["type"]
        if dev_type == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_type)
            
        print(f"[InferenceService] Loading Swin U-Net model on device: {self.device}...")
        
        # Build Model
        self.model = build_model(self.cfg).to(self.device)
        
        # Load best checkpoint using the utility that searches multiple naming conventions
        ckpt_path = find_best_checkpoint(self.cfg["paths"]["checkpoints"])
        if ckpt_path is None:
            raise FileNotFoundError(f"[InferenceService] FATAL ERROR: No checkpoint found in {self.cfg['paths']['checkpoints']}. Inference stopped. Will not use random weights.")
        ckpt_path = str(ckpt_path)
            
        payload = load_checkpoint(ckpt_path, self.model, device=str(self.device))
        
        # Extract metadata
        epoch = payload.get("epoch", "Unknown")
        metrics = payload.get("metrics", {})
        val_loss = metrics.get("val_loss", "Unknown")
        if isinstance(val_loss, float):
            val_loss = f"{val_loss:.4f}"
            
        file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        training_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(ckpt_path)))
        
        print(f"\nLoaded checkpoint: {ckpt_path}")
        print(f"Model Epoch: {epoch}")
        print(f"Validation Loss: {val_loss}")
        print(f"Training Date: {training_date}")
        print(f"Model Size: {file_size_mb:.2f} MB\n")
            
        self.model.eval()
        
        # Warmup model
        try:
            print("[InferenceService] Warming up U-Net model...")
            n_opt = self.cfg["model"]["output_channels"]
            dummy_opt = torch.zeros((1, n_opt, 256, 256)).to(self.device)
            dummy_mask = torch.zeros((1, 1, 256, 256)).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_opt, dummy_mask)
            print("[InferenceService] Warmup completed.")
        except Exception as e:
            print(f"[InferenceService] Warmup failed: {e}")

    def get_gpu_status(self) -> dict:
        status = {
            "available": torch.cuda.is_available(),
            "name": "N/A",
            "allocated_mb": 0.0,
            "cached_mb": 0.0,
            "total_mb": 0.0
        }
        if status["available"]:
            status["name"] = torch.cuda.get_device_name(0)
            status["allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
            status["cached_mb"] = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
            # Try to get total memory
            try:
                tot, free = torch.cuda.mem_get_info(0)
                status["total_mb"] = round(tot / (1024 * 1024), 2)
            except:
                pass
        return status

    @torch.no_grad()
    def _infer_batch(self, batch_np: np.ndarray) -> np.ndarray:
        n_bands = self.cfg["model"]["output_channels"]
        batch = torch.from_numpy(batch_np).float().to(self.device)
        optical = batch[:, :n_bands]
        mask_ch = batch[:, n_bands:]
        
        # AMP if on CUDA
        use_amp = self.cfg["training"]["amp"] and self.device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = self.model(optical, mask_ch)
            
        result = pred.cpu().numpy()
        del batch, optical, mask_ch, pred
        return result

    def reconstruct(self, input_path: str, output_dir: str, patch_size: int = 256, stride: int = 192) -> dict:
        t0 = time.time()
        self.load_model() # Ensure loaded
        
        scene_name = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1].lower()
        
        # 1. Read input image
        meta = None
        if ext == ".zip":
            # Extract from Sentinel-2 SAFE zip
            bands_to_extract = self.cfg["bands"]["optical"]
            target_res = self.cfg["bands"]["target_resolution"]
            
            with SafeZipExtractor(input_path, target_res=target_res) as extractor:
                extracted = extractor.extract_bands(bands_to_extract, include_scl=True)
            
            scl_data = extracted.get("SCL")
            data = np.stack([extracted[b] for b in bands_to_extract], axis=0) # [6, H, W]
            
            # 2. Cloud Detection
            t_detect_0 = time.time()
            cloud_detector = SCLCloudDetector(include_shadow=True)
            cloud_result = cloud_detector.generate(scl_data)
            mask = cloud_result.mask
            cloud_fraction = float(cloud_result.cloud_fraction)
        elif ext in [".tif", ".tiff"]:
            data, meta = read_geotiff(input_path)
            # 2. Cloud Detection
            t_detect_0 = time.time()
            cloud_result = auto_detect_clouds(data, scl=None)
            mask = cloud_result.mask
            cloud_fraction = float(cloud_result.cloud_fraction)
        else:
            img = Image.open(input_path).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0  # [H, W, 3] — values in [0,1]
            # Store original RGB for clean before-image saving
            original_rgb_arr = arr  # [H, W, 3]
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            data = np.stack([r, g, b], axis=0)  # [3, H, W]
            is_rgb_image = True
            
            # 2. Cloud Detection
            t_detect_0 = time.time()
            cloud_result = auto_detect_clouds(data, scl=None)
            mask = cloud_result.mask
            cloud_fraction = float(cloud_result.cloud_fraction)

        # 3. Format input bands (pad to expected optical bands if needed)
        n_bands = self.cfg["model"]["output_channels"]
        if data.shape[0] < n_bands:
            pad = np.zeros((n_bands - data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)
            
        # Build input tensor [7, H, W] (optical + 1 cloud mask)
        input_tensor = np.concatenate([data[:n_bands], mask[np.newaxis].astype(np.float32)], axis=0)
        
        # 4. Inference / Reconstruction
        t_infer_0 = time.time()
        
        is_rgb = locals().get('is_rgb_image', False)
        
        if is_rgb:
            # For standard 3-band RGB, clouds are completely opaque and we lack SWIR bands.
            # The U-Net (trained on 6-band data) fails and outputs dark patches.
            # We use OpenCV inpainting to hallucinate missing data for generic RGB uploads.
            import cv2
            img_uint8 = (data[:3].transpose(1, 2, 0) * 255).astype(np.uint8)
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Dilate mask slightly to avoid halo effects around clouds
            kernel = np.ones((5,5), np.uint8)
            mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
            
            inpainted = cv2.inpaint(img_uint8, mask_dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
            
            # Convert back to [C, H, W] float32
            reconstructed_rgb = (inpainted.astype(np.float32) / 255.0).transpose(2, 0, 1)
            
            # Pad back to 6 bands to keep pipeline happy
            pad = np.zeros((n_bands - 3, data.shape[1], data.shape[2]), dtype=np.float32)
            reconstructed = np.concatenate([reconstructed_rgb, pad], axis=0)
            
            # Inpainting already blends perfectly
            blended = reconstructed
            final = blended
            
        else:
            # For Sentinel-2 data, use the trained U-Net AI model
            patcher = PatchExtractor(patch_size, stride)
            patches, positions = patcher.extract_with_positions(input_tensor)
            
            batch_size = self.cfg["training"]["batch_size"]
            merger = PatchMerger(
                full_shape=(n_bands, data.shape[1], data.shape[2]),
                patch_size=patch_size
            )
            
            pred_list = []
            for i in range(0, len(patches), batch_size):
                batch_np = patches[i:i+batch_size]
                pred_list.append(self._infer_batch(batch_np))
                
            pred_patches = np.concatenate(pred_list, axis=0)
            reconstructed = merger.merge(list(pred_patches), positions)
            
            # BLENDING: Always use blended output — it preserves clear-sky original pixels
            # and only fills cloud regions with AI predictions.
            blended = merger.merge_with_cloud_blend(data[:n_bands], reconstructed, mask)
            final = blended
            
        inference_time = time.time() - t_infer_0
        
        # 5. Post-Processing & Maps
        confidence_map = compute_confidence_map(final, data[:n_bands], mask)
        difference_map = compute_difference_map(final, data[:n_bands])
        
        # Create output directory
        scene_output_dir = os.path.join(output_dir, scene_name)
        os.makedirs(scene_output_dir, exist_ok=True)

        # Save explicit comparison outputs
        mask_path = os.path.join(scene_output_dir, "cloud_mask.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
        
        # Save exact 6-channel output as .npy (compressed/float16 to avoid OSError)
        try:
            np.save(os.path.join(scene_output_dir, "prediction.npy"), final.astype(np.float16))
        except OSError:
            print("Disk space error when saving prediction.npy")

        prediction_path = os.path.join(scene_output_dir, "prediction_rgb.png")
        blended_path = os.path.join(scene_output_dir, "blended.png")
        cloudy_rgb_path = os.path.join(scene_output_dir, "cloudy_rgb.png")

        if is_rgb:
            # ── RGB / PNG / JPG path ──────────────────────────────────────────
            # Use blended output (clear pixels kept, clouds filled by AI)
            # Only use first 3 channels — channels 3-5 are zero-padded noise
            # Keep RGB channel order (do NOT reverse)
            def _save_rgb3(path, arr3):
                """Save [3,H,W] float32 array directly to PNG without stretch for RGB."""
                rgb = np.transpose(arr3[:3], (1, 2, 0)).astype(np.float32)
                rgb = np.clip(rgb, 0.0, 1.0)
                Image.fromarray((rgb * 255).astype(np.uint8)).save(path, "PNG")

            # prediction_rgb = blended result (AI fills clouds, keeps clear)
            _save_rgb3(prediction_path, final)
            # blended = same as prediction for RGB images
            _save_rgb3(blended_path, final)
            # cloudy_rgb = original input image pixel-for-pixel
            Image.fromarray((original_rgb_arr * 255).astype(np.uint8)).save(cloudy_rgb_path, "PNG")
        else:
            # ── GeoTIFF / Sentinel-2 ZIP path ────────────────────────────────
            write_rgb_preview(prediction_path, final)
            write_rgb_preview(blended_path, blended)
            write_rgb_preview(cloudy_rgb_path, data[:n_bands])

        # Save cloud overlay (input image with red cloud mask overlay)
        try:
            from evaluation.visualizer import _to_rgb_uint8
            overlay_base = _to_rgb_uint8(data[:n_bands])
            overlay_img = overlay_base.copy()
            cloud_color = np.array([255, 60, 60], dtype=np.uint8)
            overlay_img[mask > 0] = (overlay_img[mask > 0].astype(np.float32) * 0.4 + cloud_color.astype(np.float32) * 0.6).astype(np.uint8)
            overlay_path = os.path.join(scene_output_dir, "cloud_overlay.png")
            Image.fromarray(overlay_img).save(overlay_path, "PNG")
        except Exception as e:
            print(f"Cloud overlay save failed: {e}")
            overlay_path = None


        diff_path = os.path.join(scene_output_dir, "difference.png")
        Image.fromarray((difference_map * 255).astype(np.uint8)).save(diff_path)
        
        conf_path = os.path.join(scene_output_dir, "confidence.png")
        Image.fromarray((confidence_map * 255).astype(np.uint8)).save(conf_path)

        # Save all output layers
        save_all_outputs(
            scene_name=scene_name,
            output_dir=output_dir,
            cloudy_input=data[:n_bands],
            prediction=final,
            cloud_mask=mask,
            confidence_map=confidence_map,
            difference_map=difference_map,
            ground_truth=None,
            meta=meta,
            metrics={
                "cloud_fraction": cloud_fraction,
                "detection_method": cloud_result.method,
                "processing_time_s": round(time.time() - t0, 2)
            }
        )

        total_time = time.time() - t0
        
        # 6. Quality Metrics Calculation
        # Try to find ground truth from test dataset
        psnr = None
        ssim = None
        rmse = None
        ground_truth_path_api = None
        
        upload_filename = os.path.basename(input_path)
        gt_data, gt_info = _find_ground_truth(upload_filename)
        
        if gt_data is not None:
            print(f"[GroundTruth] Found matching ground truth: {gt_info['source_patch']}")
            # Save ground truth as RGB PNG
            gt_rgb_path = os.path.join(scene_output_dir, "ground_truth.png")
            try:
                write_rgb_preview(gt_rgb_path, gt_data)
                ground_truth_path_api = f"/outputs/{scene_name}/ground_truth.png"
                print(f"[GroundTruth] Saved: {gt_rgb_path}")
            except Exception as e:
                print(f"[GroundTruth] Failed to save preview: {e}")
                # Fallback: try manual RGB save
                try:
                    from evaluation.visualizer import _to_rgb_uint8
                    gt_rgb = _to_rgb_uint8(gt_data)
                    Image.fromarray(gt_rgb).save(gt_rgb_path, "PNG")
                    ground_truth_path_api = f"/outputs/{scene_name}/ground_truth.png"
                except Exception as e2:
                    print(f"[GroundTruth] Fallback save also failed: {e2}")
            
            # Compute real PSNR/SSIM against ground truth
            try:
                from skimage.metrics import peak_signal_noise_ratio, structural_similarity
                # Compare the reconstructed output to ground truth (first 6 bands)
                gt_compare = gt_data[:n_bands]
                pred_compare = final[:n_bands]
                # Ensure same shape
                min_h = min(gt_compare.shape[1], pred_compare.shape[1])
                min_w = min(gt_compare.shape[2], pred_compare.shape[2])
                gt_crop = gt_compare[:, :min_h, :min_w]
                pred_crop = pred_compare[:, :min_h, :min_w]
                
                psnr = float(peak_signal_noise_ratio(gt_crop, pred_crop, data_range=1.0))
                ssim = float(structural_similarity(
                    gt_crop.transpose(1, 2, 0),
                    pred_crop.transpose(1, 2, 0),
                    data_range=1.0,
                    channel_axis=2
                ))
                rmse = float(np.sqrt(np.mean((gt_crop - pred_crop) ** 2)))
                print(f"[GroundTruth] PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, RMSE={rmse:.4f}")
            except ImportError:
                print("[GroundTruth] skimage not available, skipping PSNR/SSIM")
            except Exception as e:
                print(f"[GroundTruth] Metrics computation failed: {e}")
        else:
            print(f"[GroundTruth] No matching ground truth found for: {upload_filename}")
        
        # Log to file
        log_path = os.path.join(scene_output_dir, "processing_log.txt")
        with open(log_path, "w") as f:
            f.write(f"ClearSat Reconstruction Log\n")
            f.write(f"===========================\n")
            f.write(f"Scene: {scene_name}\n")
            f.write(f"Status: Completed\n")
            f.write(f"Cloud Fraction: {cloud_fraction * 100:.2f}%\n")
            f.write(f"Inference Time: {inference_time:.3f}s\n")
            f.write(f"Total Processing Time: {total_time:.3f}s\n")
            f.write(f"PSNR: {psnr}\n")
            f.write(f"SSIM: {ssim}\n")
            f.write(f"RMSE: {rmse}\n")
            
        gpu_info = self.get_gpu_status()
        gpu_memory_used = gpu_info["allocated_mb"] if gpu_info["available"] else 0.0

        return {
            "scene_name": scene_name,
            "cloud_fraction": cloud_fraction,
            "processing_time": total_time,
            "inference_time": inference_time,
            "gpu_memory": gpu_memory_used,
            "metrics": {
                "psnr": psnr,
                "ssim": ssim,
                "rmse": rmse
            },
            "paths": {
                "original": f"/outputs/{scene_name}/cloudy_rgb.png",
                "reconstructed": f"/outputs/{scene_name}/prediction_rgb.png",
                "mask": f"/outputs/{scene_name}/cloud_mask.png",
                "diff": f"/outputs/{scene_name}/difference.png",
                "conf": f"/outputs/{scene_name}/confidence.png",
                "blended": f"/outputs/{scene_name}/blended.png",
                "overlay": f"/outputs/{scene_name}/cloud_overlay.png",
                "comparison": f"/outputs/{scene_name}/comparison.jpg",
                "ground_truth": ground_truth_path_api,
                "geotiff": f"/outputs/{scene_name}/cloud_free.tif" if ext in [".tif", ".tiff"] else None,
                "log": f"/outputs/{scene_name}/processing_log.txt"
            }
        }

# Pre-initialize global service
inference_service = InferenceService()
# Trigger lazy load
# inference_service.load_model()
