import os

class Settings:
    PROJECT_NAME: str = "ClearSat"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "clearsat_super_secret_jwt_key_2026_meghamukt")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 # 24 hours

    # Absolute paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
    
    CHECKPOINT_PATH = os.path.join(
        os.path.dirname(BASE_DIR), 
        "cloud-reconstruction", 
        "checkpoints_6band", 
        "best.pth"
    )
    CONFIG_YAML_PATH = os.path.join(
        os.path.dirname(BASE_DIR), 
        "cloud-reconstruction", 
        "config.yaml"
    )

    def __init__(self):
        os.makedirs(self.UPLOADS_DIR, exist_ok=True)
        os.makedirs(self.OUTPUTS_DIR, exist_ok=True)
        os.makedirs(self.DATASETS_DIR, exist_ok=True)

settings = Settings()
