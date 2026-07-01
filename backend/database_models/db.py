from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text, Boolean
from sqlalchemy.orm import relationship
import datetime
from database.connection import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    uploads = relationship("Upload", back_populates="user")
    reconstructions = relationship("Reconstruction", back_populates="user")
    settings = relationship("Setting", back_populates="user", uselist=False)

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False) # LISS-IV, Sentinel-1, Sentinel-2, DEM
    resolution = Column(String)
    crs = Column(String)
    bands = Column(String)
    satellite = Column(String)
    acquisition_date = Column(String)
    file_path = Column(String, nullable=False)
    file_size = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Upload(Base):
    __tablename__ = "uploads"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    filename = Column(String, nullable=False)
    file_size = Column(Float)
    resolution = Column(String)
    width = Column(Integer)
    height = Column(Integer)
    bands = Column(String)
    crs = Column(String)
    acquisition_date = Column(String)
    satellite_name = Column(String)
    file_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="uploads")
    reconstruction = relationship("Reconstruction", back_populates="upload")

class Reconstruction(Base):
    __tablename__ = "reconstructions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    upload_id = Column(Integer, ForeignKey("uploads.id"), nullable=False)
    status = Column(String, default="pending") # pending, validating, processing, completed, failed
    error_message = Column(String, nullable=True)
    cloud_fraction = Column(Float, default=0.0)
    processing_time = Column(Float, default=0.0)
    inference_time = Column(Float, default=0.0)
    gpu_memory = Column(Float, default=0.0)
    original_path = Column(String)
    reconstructed_path = Column(String)
    mask_path = Column(String)
    diff_path = Column(String)
    conf_path = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="reconstructions")
    upload = relationship("Upload", back_populates="reconstruction")
    metrics = relationship("Metric", back_populates="reconstruction", uselist=False)

class Metric(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True, index=True)
    reconstruction_id = Column(Integer, ForeignKey("reconstructions.id"), nullable=False)
    psnr = Column(Float)
    ssim = Column(Float)
    rmse = Column(Float)
    cloud_coverage = Column(Float)
    processing_time = Column(Float)
    inference_time = Column(Float)
    gpu_memory = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    reconstruction = relationship("Reconstruction", back_populates="metrics")

class Setting(Base):
    __tablename__ = "settings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, unique=True)
    theme = Column(String, default="dark")
    language = Column(String, default="en")
    device = Column(String, default="auto") # cuda, cpu, auto
    batch_size = Column(Integer, default=8)
    patch_size = Column(Integer, default=256)
    inference_mode = Column(String, default="Standard")

    user = relationship("User", back_populates="settings")

class Log(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True, index=True)
    level = Column(String, default="info")
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
