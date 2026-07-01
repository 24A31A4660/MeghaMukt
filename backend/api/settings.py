from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.connection import get_db
from database_models.db import Setting, User
from api.auth import get_optional_user
from schemas.api import SettingUpdate, SettingResponse

router = APIRouter(prefix="/api/settings", tags=["settings"])

@router.get("", response_model=SettingResponse)
def get_settings(
    db: Session = Depends(get_db),
    user: User = Depends(get_optional_user)
):
    if user:
        setting = db.query(Setting).filter(Setting.user_id == user.id).first()
        if not setting:
            setting = Setting(user_id=user.id)
            db.add(setting)
            db.commit()
            db.refresh(setting)
    else:
        # Default settings for Guest Mode
        return {
            "theme": "dark",
            "language": "en",
            "device": "auto",
            "batch_size": 8,
            "patch_size": 256,
            "inference_mode": "Standard"
        }
        
    return {
        "theme": setting.theme,
        "language": setting.language,
        "device": setting.device,
        "batch_size": setting.batch_size,
        "patch_size": setting.patch_size,
        "inference_mode": setting.inference_mode
    }

@router.put("", response_model=SettingResponse)
def update_settings(
    settings_data: SettingUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_optional_user)
):
    if not user:
        # For guests, just echo the updated settings back (session-only)
        return settings_data
        
    setting = db.query(Setting).filter(Setting.user_id == user.id).first()
    if not setting:
        setting = Setting(user_id=user.id)
        db.add(setting)
        
    if settings_data.theme is not None:
        setting.theme = settings_data.theme
    if settings_data.language is not None:
        setting.language = settings_data.language
    if settings_data.device is not None:
        setting.device = settings_data.device
    if settings_data.batch_size is not None:
        setting.batch_size = settings_data.batch_size
    if settings_data.patch_size is not None:
        setting.patch_size = settings_data.patch_size
    if settings_data.inference_mode is not None:
        setting.inference_mode = settings_data.inference_mode
        
    db.commit()
    db.refresh(setting)
    
    return {
        "theme": setting.theme,
        "language": setting.language,
        "device": setting.device,
        "batch_size": setting.batch_size,
        "patch_size": setting.patch_size,
        "inference_mode": setting.inference_mode
    }
