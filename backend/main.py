import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ── Path Setup ──────────────────────────────────────────────────────────────
# This file lives in backend/. PROJECT_ROOT is one level up.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

# ── Internal imports (order matters — config before DB models) ───────────────
from core.config import settings as config_settings
from database.connection import engine, Base
from api import auth, processing, results, settings, datasets
from services.inference_service import inference_service

# ── Create all SQLite tables ─────────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

# ── Lifespan (modern FastAPI startup/shutdown) ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- STARTUP ----
    print("[ClearSat] Starting FastAPI server...")
    try:
        inference_service.load_model()
        print("[ClearSat] U-Net model ready in GPU/CPU memory.")
    except Exception as e:
        print(f"[ClearSat] Model load failed (non-fatal): {e}")
    yield
    # ---- SHUTDOWN ----
    print("[ClearSat] Shutting down.")

# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="ClearSat — AI-Powered Satellite Imagery Reconstruction",
    description="ISRO Bharatiya Antariksh Hackathon 2026 — Team Meghamukt",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routers ───────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(processing.router)
app.include_router(results.router)
app.include_router(settings.router)
app.include_router(datasets.router)

# ── Static file mounts ────────────────────────────────────────────────────────
# Serve reconstructed output images (cloud-free RGB, masks, etc.)
app.mount("/outputs", StaticFiles(directory=config_settings.OUTPUTS_DIR), name="outputs")

# Serve uploaded satellite files for preview
app.mount("/uploads", StaticFiles(directory=config_settings.UPLOADS_DIR), name="uploads")

# Serve the original landing page's static assets
app.mount("/js",     StaticFiles(directory=os.path.join(PROJECT_ROOT, "js")),     name="js")
app.mount("/css",    StaticFiles(directory=os.path.join(PROJECT_ROOT, "css")),    name="css")
app.mount("/assets", StaticFiles(directory=os.path.join(PROJECT_ROOT, "assets")), name="assets")

# ── HTML Page Routes ──────────────────────────────────────────────────────────
@app.get("/")
@app.get("/login")
@app.get("/signup")
@app.get("/dashboard")
@app.get("/uploader.html")
def spa_routes():
    """Serve the Single Page Application (Three.js + React Dashboard)."""
    return FileResponse(os.path.join(PROJECT_ROOT, "index.html"))

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[BASE_DIR],
    )
