# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mangrove Detection Platform v2.0 — a web-based satellite image analysis platform for mangrove and vegetation detection using Sentinel-2, Sentinel-1, and EMIT imagery via Google Earth Engine.

## Running the Application

```bash
# Linux/Mac (creates venv, installs deps, starts server)
bash run.sh

# Manual start (if venv already set up)
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Server runs at `http://localhost:8000`. Frontend is served as static files at `/static/`.

### Required Environment Variables

Set automatically by `run.sh`, or manually:
- `EE_SERVICE_ACCOUNT_KEY` — path to GEE service account JSON (default: `backend/ee-service-account-key.json`)
- `GCS_BUCKET` — Google Cloud Storage bucket name
- `GCS_BUCKET_REGION` — GCS region (default: `asia-northeast3`)

### Installing Dependencies

```bash
pip install -r backend/requirements.txt
```

## Architecture

**Backend:** FastAPI (Python) — `backend/main.py` is the entry point. Uses `uvicorn` with `--reload`.

**Frontend:** Vanilla JavaScript + Leaflet.js (no build step, no bundler). Loaded via CDN dependencies. Served as static files by FastAPI.

### Backend Structure

- `backend/api/` — FastAPI route handlers, split by domain:
  - `routes_search.py` — satellite image search
  - `routes_process.py` — image download and mosaic processing
  - `routes_analysis.py` — spectral analysis and visualization
  - `routes_download.py` — file download endpoints
  - `routes_target_detection.py` — hyperspectral target detection
  - `schemas.py` — Pydantic request/response models
- `backend/services/` — business logic:
  - `earth_engine.py` — GEE initialization and collection management
  - `downloader.py` — parallel tile download and mosaic generation
  - `model_inference.py` — Segformer/UNet++ PyTorch segmentation (patch-based, optional)
  - `spectral_analysis.py` — NDVI, NDMI, MVI calculations
  - `target_detection.py` — SAM, ACE, RXD, CEM algorithms
  - `visualization.py` — image rendering, band stretching, PNG generation
- `backend/core/config.py` — paths, constants, GEE credentials, HTTP pooling, band definitions
- `backend/core/progress.py` — thread-safe job progress tracking with phase weights
- `backend/model_config.yaml` — segmentation model configuration

### Frontend Structure

- `frontend/index.html` — single-page app shell
- `frontend/script.js` — main controller, initializes all modules
- `frontend/map.js` — Leaflet map setup
- `frontend/js/` — modular components:
  - `api-client.js` — HTTP wrapper for all `/api/` calls
  - `map-core.js`, `map-layers.js`, `map-drawing.js`, `map-file-loader.js` — map functionality
  - `image-search.js`, `image-processor.js`, `search-results.js` — search/process UI
  - `analysis-controller.js`, `threshold-controller.js` — spectral analysis UI
  - `target-detection.js` — target detection UI
  - `change-monitoring.js` — time-series monitoring with calendar

### Data Flow

1. User draws AOI on Leaflet map
2. Frontend calls `/api/search-images` to find available satellite images
3. User selects image, frontend calls `/api/process-image`
4. Backend downloads tiles from GEE in parallel, creates mosaic, saves to `outputs/`
5. Frontend polls `/api/progress` for job status
6. Results served via `/api/proxy-file` (supports HTTP Range requests)
7. Frontend displays results as Leaflet map layers

### Key Design Patterns

- **Progress tracking:** Long-running tasks use job IDs with phase-weighted progress, polled via `/api/progress`
- **File serving:** `/api/proxy-file` streams files from `outputs/` directory with Range request support for large GeoTIFFs
- **Model graceful degradation:** Segmentation model is optional; platform works without it for spectral analysis and visualization
- **Parallel downloads:** Mosaic builder uses 6 parallel workers for tile downloads from GEE
- **In-memory caching:** Raster data, spectral indices, and visualization results are cached

## Notes

- No test suite exists in this project
- Documentation (`docs/FUNCTIONAL_SPECIFICATION.md`) is in Korean
- CORS is open (`allow_origins=["*"]`)
- The `ee-service-account-key.json` file is required but git-ignored
