from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str

class UserProfile(BaseModel):
    username: str
    email: EmailStr

class SettingUpdate(BaseModel):
    theme: Optional[str] = "dark"
    language: Optional[str] = "en"
    device: Optional[str] = "auto"
    batch_size: Optional[int] = 8
    patch_size: Optional[int] = 256
    inference_mode: Optional[str] = "Standard"

class SettingResponse(BaseModel):
    theme: str
    language: str
    device: str
    batch_size: int
    patch_size: int
    inference_mode: str

class UploadMetadata(BaseModel):
    id: int
    filename: str
    file_size_mb: float
    resolution: str
    width: int
    height: int
    bands: List[str]
    crs: str
    acquisition_date: str
    satellite_name: str

class ReconstructionMetrics(BaseModel):
    psnr: float
    ssim: float
    rmse: float
    cloud_fraction: float
    processing_time: float
    inference_time: float
    gpu_memory: float

class ReconstructionResponse(BaseModel):
    id: int
    status: str
    cloud_fraction: float
    processing_time: float
    inference_time: float
    gpu_memory: float
    original_path: str
    reconstructed_path: str
    mask_path: str
    diff_path: str
    conf_path: str
    metrics: Optional[ReconstructionMetrics] = None
