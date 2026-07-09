# MeghaMukt: AI-Powered Satellite Cloud Removal

## Project Overview
MeghaMukt is a scalable, automated pipeline leveraging Generative AI to reconstruct cloud-covered regions in satellite imagery. Designed specifically for India's North Eastern Region (NER), this tool restores high-fidelity, analysis-ready multi-spectral data crucial for disaster response, Land Use / Land Cover (LULC) mapping, and infrastructure monitoring.

## Problem Statement
**PS-2 — Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery**

*   **Live Demo:** [https://meghamukt.vercel.app/](https://meghamukt.vercel.app/)
*   **Presentation PDF:** [Bah Hackathon.pdf](docs/Bah_Hackathon.pdf)

NER of India experiences cloud cover 6–8 months per year due to persistent monsoon activity. Only 10–15% of LISS-IV acquisitions over NER are cloud-free, creating critical data gaps during floods and landslides. Traditional masking discards valuable data, while standard inpainting lacks the spectral fidelity required for scientific analysis. MeghaMukt solves this by employing Latent Diffusion Models (LDM) conditioned on cloud-penetrating Sentinel-1 SAR imagery.

## Features
*   **SAR-Guided Diffusion**: Latent Diffusion Model conditioned on Sentinel-1 SAR data provides cloud-free structural priors for any cloud density.
*   **Multi-Spectral Fidelity**: Spectral Angle Mapper (SAM) loss forces band-wise consistency so reconstructed imagery is analysis-ready.
*   **Automated Pipeline**: End-to-end workflow from Bhoonidhi ingest to GeoTIFF export with quality metrics—zero manual steps.
*   **Transfer Learning**: Pretrained on SEN12MS-CR (Sentinel-2 cloud removal) then fine-tuned on LISS-IV for fast convergence.
*   **Temporal Fusion**: Temporal reference imagery from prior cloud-free acquisitions provides additional context for ambiguous regions.
*   **Scalable Deployment**: Docker-containerised REST API supports on-demand and batch cloud removal for operational satellite workflows.

## Architecture
**DATA LAYER**
LISS-IV (Bhoonidhi) | Sentinel-1 SAR (Copernicus) | Sentinel-2 MSI | SRTM DEM | Temporal Reference Imagery
?
**PREPROCESSING LAYER**
Cloud Masking (Fmask + U-Net) ? Co-registration (GDAL) ? Radiometric Normalisation ? Patch Extraction (Albumentations)
?
**MODEL LAYER**
VQ-VAE Encoder ? Dual-path SAR+Mask Conditioning ? U-Net Denoiser (Cross-attention) ? VQ-VAE Decoder
?
**OUTPUT LAYER**
Cloud-free GeoTIFF | Quality Flags (PSNR/SSIM) | LULC / Disaster Monitoring / Infrastructure Analysis

## AI Pipeline
**Process Flow:**
1.  **Data Acquisition**: Download LISS-IV from Bhoonidhi portal. Acquire Sentinel-1 SAR (IW mode, VV/VH). Download DEM and Sentinel-2 for pretraining.
2.  **Preprocessing**: Cloud masking with Fmask 4.0 + U-Net refinement. Co-registration (GDAL gdalwarp), TOA reflectance correction. Patch extraction: 256×256 tiles at 128px stride with Albumentations augmentation.
3.  **Model Training & Inference**: Pretrain LDM on SEN12MS-CR, then fine-tune on LISS-IV pairs. Combined loss function: L1 + SSIM + Perceptual (VGG-19) + SAM spectral consistency. Batch inference ? Poisson-blend patches ? export cloud-free GeoTIFF.

## Technology Stack
*   **Deep Learning**: Python 3.10+, PyTorch 2.x, Hugging Face Diffusers, ONNX Runtime, Weights & Biases
*   **Computer Vision**: OpenCV, Scikit-image, Albumentations
*   **Geospatial Tools**: GDAL 3.x, Rasterio, QGIS, Google Earth Engine, Folium
*   **Deployment**: Docker, FastAPI, Node.js (Frontend Dashboard)
*   **Hardware**: GPU: NVIDIA A100 / V100 / RTX 4050

## Dataset
*   **LISS-IV**: Cloudy + cloud-free pairs from the Bhoonidhi portal (3 bands: Green, Red, NIR, 5.8m resolution).
*   **Sentinel-1 SAR**: Cloud-penetrating C-band synthetic aperture radar (Copernicus Open Access Hub).
*   **SEN12MS-CR dataset**: Used for pretraining the foundation model before fine-tuning on LISS-IV data.

## Model
**SAR-Conditioned Latent Diffusion Model & Swin U-Net**
*   **Inputs**:
    *   Channel 1 — Cloudy LISS-IV patch (256×256 pixels)
    *   Channel 2 — Cloud mask (binary mask, 1=cloud, 0=clear)
    *   Channel 3 — Sentinel-1 SAR patch (VV + VH polarisation)
*   **Architecture**: VQ-VAE encoder compresses to 32×32×4 latent space. Dual-path conditioning encoder fuses SAR + mask embeddings. U-Net denoiser with cross-attention.
*   **Output**: Cloud-free 3-band LISS-IV patch with original georeference, per-pixel confidence map, and PSNR/SSIM quality flags.

## Results
MeghaMukt outperforms prior GANs by **3–5 dB PSNR** on thick cloud benchmarks, providing analysis-ready scientific imagery rather than just visually plausible inpainting. The Spectral Angle Mapper (SAM) loss ensures that NDVI and LULC calculations remain valid on the reconstructed pixels.

## Dashboard
The project includes a sleek, interactive frontend dashboard powered by React and Node.js. It allows users to:
*   Upload cloudy LISS-IV satellite imagery.
*   View the AI-generated cloud mask.
*   Preview the SAR-conditioned cloud-free reconstruction.
*   Download the final GeoTIFF and analysis metrics.

## Installation
1.  Clone this repository.
2.  Install Python dependencies for the AI model: pip install -r cloud-reconstruction/requirements.txt
3.  Install Node dependencies for the backend and dashboard: cd backend && npm install
4.  Ensure cloud-reconstruction/dataset and cloud-reconstruction/checkpoints_swin are populated with your data and weights.

## Usage
1.  Start the AI Backend: cd backend && npm start
2.  Open your browser and navigate to http://localhost:8000/dashboard
3.  Upload your satellite images to trigger the pipeline.
4.  To run headless training or evaluation, use python cloud-reconstruction/train_optimized.py.

## Folder Structure
*   `backend/`: Node.js Express server to handle uploads and interface with the AI Python scripts.
*   `cloud-reconstruction/`: The core AI Python pipeline (preprocessing, models, training, evaluation).
*   `frontend/`: Frontend web components for the interactive dashboard (HTML, CSS, JS, assets).
*   `docs/`: Additional documentation and forensic verification data.
*   `scripts/`: Utility shell scripts.
*   `sample_data/`: 2-5 small example patches (optional).
*   `requirements.txt`: Python pip dependencies.
*   `environment.yml`: Conda environment definition.

## 👥 Team

| Member | Role | GitHub |
|---------|------|--------|
| **B.N.V. Chaitanya Yadav** | Team Leader, Project Planning & System Integration | [@bogulachaitanya](https://github.com/bogulachaitanya) |
| **B.N.L. Niharika** | Data Preprocessing, Cloud Mask Generation & Validation | [@bogulaniharika](https://github.com/bogulaniharika) |
| **G. Akshaya** | Research, Documentation & Presentation | [@akshayagajjela](https://github.com/akshayagajjela) |
| **T. Akshay Kountesh** | AI Model Development, Swin U-Net, Backend, Dashboard & Deployment | [@24A31A4660](https://github.com/24A31A4660) |

## 🤝 Contributors

This project was developed collaboratively by Team **MeghaMukt** for the **Bharatiya Antariksh Hackathon 2026 (Problem Statement PS-2)**.

Special contributions include:

- **B.N.V. Chaitanya Yadav**
  - Team Leadership
  - Project Planning
  - System Architecture
  - Team Coordination

- **B.N.L. Niharika**
  - Dataset Preparation
  - Data Preprocessing
  - Cloud Detection & Validation
  - Testing

- **G. Akshaya**
  - Research
  - Literature Survey
  - Documentation
  - Presentation Design

- **T. Akshay Kountesh**
  - AI Model Development
  - Swin U-Net Implementation
  - Model Training & Evaluation
  - Backend Development
  - Dashboard Development
  - GitHub Repository Management
  - Deployment & Integration

- B.N.V. Chaitanya Yadav
  [![GitHub](https://img.shields.io/badge/GitHub-bogulachaitanya-181717?logo=github)](https://github.com/bogulachaitanya)

- B.N.L. Niharika
  [![GitHub](https://img.shields.io/badge/GitHub-bogulaniharika-181717?logo=github)](https://github.com/bogulaniharika)

- G. Akshaya
  [![GitHub](https://img.shields.io/badge/GitHub-akshayagajjela-181717?logo=github)](https://github.com/akshayagajjela)

- T. Akshay Kountesh
  [![GitHub](https://img.shields.io/badge/GitHub-24A31A4660-181717?logo=github)](https://github.com/24A31A4660)

## Future Improvements
*   Implement end-to-end Dockerization for a fully portable microservices architecture.
*   Integrate temporal fusion using more advanced recurrent attention mechanisms across historical acquisitions.
*   Explore continuous on-device learning for edge-deployment in ground stations.

## License
MIT License. See LICENSE for details.
