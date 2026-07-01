import os
import time
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from database.connection import get_db
from database_models.db import Upload, Reconstruction, Metric, User
from api.auth import get_optional_user
from services.geospatial_service import GeospatialService
from services.inference_service import inference_service
from core.config import settings as config_settings

router = APIRouter(prefix="/api", tags=["processing"])

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user)
):
    # Create unique filename
    timestamp = int(time.time())
    safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
    file_path = os.path.join(config_settings.UPLOADS_DIR, safe_filename)
    
    # Save file
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
    # Validate image
    is_valid, validation_msg = GeospatialService.validate_image(file_path)
    if not is_valid:
        # Clean up invalid file
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=validation_msg)
        
    # Extract metadata
    meta = GeospatialService.get_metadata(file_path)
    
    # Save to database
    db_upload = Upload(
        user_id=user.id if user else None,
        filename=file.filename,
        file_size=meta["file_size_mb"],
        resolution=meta["resolution"],
        width=meta["width"],
        height=meta["height"],
        bands=",".join(meta["bands"]),
        crs=meta["crs"],
        acquisition_date=meta["acquisition_date"],
        satellite_name=meta["satellite"],
        file_path=file_path
    )
    db.add(db_upload)
    db.commit()
    db.refresh(db_upload)
    
    return {
        "upload_id": db_upload.id,
        "filename": db_upload.filename,
        "file_size_mb": db_upload.file_size,
        "resolution": db_upload.resolution,
        "width": db_upload.width,
        "height": db_upload.height,
        "bands": meta["bands"],
        "crs": db_upload.crs,
        "acquisition_date": db_upload.acquisition_date,
        "satellite_name": db_upload.satellite_name
    }

@router.post("/reconstruct")
def reconstruct_image(
    upload_id: int = Form(...),
    patch_size: Optional[int] = Form(256),
    stride: Optional[int] = Form(192),
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user)
):
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        raise HTTPException(status_code=404, detail="Uploaded file not found")
        
    # Create initial reconstruction record
    scene_name = os.path.splitext(os.path.basename(upload.file_path))[0]
    reconstruction = Reconstruction(
        user_id=user.id if user else None,
        upload_id=upload.id,
        status="processing",
        original_path=f"/outputs/{scene_name}/cloudy_rgb.png"
    )
    db.add(reconstruction)
    db.commit()
    db.refresh(reconstruction)
    
    try:
        # Run inference
        result = inference_service.reconstruct(
            input_path=upload.file_path,
            output_dir=config_settings.OUTPUTS_DIR,
            patch_size=patch_size,
            stride=stride
        )
        
        # Helper to safely coerce metric values to float or None
        def safe_float(v):
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        
        # Update reconstruction record
        reconstruction.status = "completed"
        reconstruction.cloud_fraction = result["cloud_fraction"]
        reconstruction.processing_time = result["processing_time"]
        reconstruction.inference_time = result["inference_time"]
        reconstruction.gpu_memory = result["gpu_memory"]
        reconstruction.reconstructed_path = result["paths"]["reconstructed"]
        reconstruction.mask_path = result["paths"]["mask"]
        reconstruction.diff_path = result["paths"]["diff"]
        reconstruction.conf_path = result["paths"]["conf"]
        
        # Add metrics record — use None when ground truth not available
        metrics = Metric(
            reconstruction_id=reconstruction.id,
            psnr=safe_float(result["metrics"]["psnr"]),
            ssim=safe_float(result["metrics"]["ssim"]),
            rmse=safe_float(result["metrics"]["rmse"]),
            cloud_coverage=result["cloud_fraction"] * 100,
            processing_time=result["processing_time"],
            inference_time=result["inference_time"],
            gpu_memory=result["gpu_memory"]
        )
        db.add(metrics)
        db.commit()
        
        return {
            "reconstruction_id": reconstruction.id,
            "status": reconstruction.status,
            "cloud_fraction": reconstruction.cloud_fraction,
            "processing_time": reconstruction.processing_time,
            "inference_time": reconstruction.inference_time,
            "gpu_memory": reconstruction.gpu_memory,
            "paths": result["paths"],
            "metrics": {
                "psnr": safe_float(result["metrics"]["psnr"]),
                "ssim": safe_float(result["metrics"]["ssim"]),
                "rmse": safe_float(result["metrics"]["rmse"])
            }
        }
        
    except Exception as e:
        reconstruction.status = "failed"
        reconstruction.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"AI Reconstruction failed: {str(e)}")


@router.post("/validate")
def validate_file(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    is_valid, msg = GeospatialService.validate_image(file_path)
    return {"valid": is_valid, "message": msg}

@router.get("/status")
def get_system_status():
    gpu_info = inference_service.get_gpu_status()
    # Estimate CPU usage & status
    import psutil
    cpu_percent = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    
    # Model status
    model_loaded = inference_service.model is not None
    
    return {
        "system": {
            "cpu_usage_percent": cpu_percent,
            "ram_used_gb": round(ram.used / (1024 * 1024 * 1024), 2),
            "ram_total_gb": round(ram.total / (1024 * 1024 * 1024), 2)
        },
        "gpu": gpu_info,
        "model": {
            "name": "U-Net Satellite Cloud Reconstruction Model",
            "architecture": "U-Net with Multi-spectral & Cloud Mask Channels",
            "status": "Ready / Resident in Memory" if model_loaded else "Standby (Lazy loading)"
        }
    }
