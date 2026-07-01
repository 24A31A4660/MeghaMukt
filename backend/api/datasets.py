from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.connection import get_db
from database_models.db import Dataset
from typing import List

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

@router.get("")
def get_datasets(db: Session = Depends(get_db)):
    # Check if empty, seed some datasets if so
    count = db.query(Dataset).count()
    if count == 0:
        seed_datasets = [
            Dataset(
                name="LISS-IV Ahmedabad Agricultural Zone",
                type="LISS-IV",
                resolution="5.8m",
                crs="EPSG:32643",
                bands="Green, Red, NIR, SWIR",
                satellite="Resourcesat-2",
                acquisition_date="2026-03-12",
                file_path="mock_liss4_ahmedabad.tif",
                file_size=120.5
            ),
            Dataset(
                name="Sentinel-2 Bengaluru Urban Coverage",
                type="Sentinel-2",
                resolution="10m",
                crs="EPSG:32643",
                bands="B02, B03, B04, B08, B11, B12",
                satellite="Sentinel-2B",
                acquisition_date="2026-05-20",
                file_path="mock_s2_bengaluru.safe.zip",
                file_size=450.2
            ),
            Dataset(
                name="Sentinel-1 Mumbai Flood Radar Data",
                type="Sentinel-1",
                resolution="20m",
                crs="EPSG:32643",
                bands="VV, VH",
                satellite="Sentinel-1A",
                acquisition_date="2026-06-01",
                file_path="mock_s1_mumbai.tif",
                file_size=88.4
            ),
            Dataset(
                name="Western Ghats Digital Elevation Model",
                type="DEM",
                resolution="30m",
                crs="EPSG:4326",
                bands="Elevation",
                satellite="SRTM",
                acquisition_date="2000-02-11",
                file_path="mock_srtm_western_ghats.tif",
                file_size=32.1
            )
        ]
        for ds in seed_datasets:
            db.add(ds)
        db.commit()
        
    datasets = db.query(Dataset).all()
    return datasets

@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.delete(ds)
    db.commit()
    return {"success": True, "message": "Dataset deleted successfully"}
