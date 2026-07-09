# MeghaMukt: AI-Powered Satellite Cloud Removal

**Problem Statement (PS-2):** Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery

MeghaMukt is a scalable, automated pipeline leveraging Generative AI (Swin U-Net architectures and Latent Diffusion Models conditioned on SAR) to reconstruct cloud-covered regions in satellite imagery. This restores high-fidelity, analysis-ready multi-spectral data crucial for disaster response and Land Use / Land Cover (LULC) mapping, particularly in persistently cloudy regions like India's North Eastern Region (NER).

## Team Members
*   **B.N.V CHAITANYA YADAV** (Team Leader) – Pragati Engineering College, Kakinada
*   **B.N.L NIHARIKA** – Pragati Engineering College, Kakinada
*   **G. AKSHAYA** – Jawaharlal Nehru Technological University, Kakinada
*   **T. AKSHAY KOUNTESH** – Pragati Engineering College, Kakinada

## Core Features
1.  **SAR-Guided Diffusion / Swin U-Net Processing:** Uses cloud-penetrating Sentinel-1 SAR and robust U-Net models to provide cloud-free structural priors.
2.  **Multi-Spectral Fidelity:** Employs Spectral Angle Mapper (SAM) loss ensuring output imagery is not just visually filled, but scientifically analysis-ready.
3.  **Automated Pipeline:** End-to-end workflow from ingest to GeoTIFF export with full quality metrics.

## Getting Started
1.  Install Python dependencies and Node modules.
2.  Start the Node.js backend (
pm start).
3.  Open the web dashboard in your browser.
