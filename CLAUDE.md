# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EarthScope v2.0 — a web-based satellite image analysis platform for Earth observation (vegetation, mangrove, target detection, change detection, segmentation) using Sentinel-2, Sentinel-1, and EMIT imagery via Google Earth Engine.

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

All deployment-specific settings live in `.env` at the project root (git-ignored;
`.env.example` is the tracked template). `backend/core/config.py` loads it through
`python-dotenv`, and `run.sh` / `run.bat` / `scripts/finish_sam3_setup.sh` source it
so child processes and the Google client libraries see the same values. Real
environment variables take precedence over the file.

- `EE_SERVICE_ACCOUNT_KEY` — path to GEE service account JSON (default: `backend/ee-service-account-key.json`; relative paths resolve against the project root)
- `EE_SERVICE_ACCOUNT` — service account address; when empty it is read from `client_email` in the key file
- `GCS_BUCKET` / `GCS_BUCKET_REGION` — optional, only for server-side GeoTIFF export of large AOIs
- `SAM3_CHECKPOINT_DIR`, `SAM3_BPE_PATH`, `MODEL_ROOT`, `LOCAL_BASE_DIR` — optional path overrides
- `HTTP_POOL`, `PARALLEL_TILE_WORKERS` — download tuning

Do not hardcode deployment values in source; add them to `.env.example` instead.
`docs/` is git-ignored (local working notes) — user-facing documentation is `README.md`.

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
  - `routes_sam3.py` — SAM3 segmentation (point/box + text/PCS modes)
  - `routes_mangrove_segmentation.py` — Segformer/UNet++ inference
  - `routes_change_detection.py` — Time A vs Time B mask comparison
  - `routes_local.py` — uploaded GeoTIFF endpoints (mirrors of all cloud features)
  - `schemas.py` — Pydantic request/response models
- `backend/services/` — business logic:
  - `earth_engine.py` — GEE initialization and collection management
  - `downloader.py` — parallel tile download and mosaic generation
  - `model_inference.py` — Segformer/UNet++ PyTorch segmentation (patch-based, optional)
  - `spectral_analysis.py` — NDVI, NDMI, MVI calculations
  - `target_detection.py` — SAM, ACE, RXD, CEM algorithms
  - `sam3_service.py` — SAM3 model loader & encode/predict/predict_text wrappers
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
- **Earth Engine graceful degradation:** `init_earth_engine()` records failures instead of raising, so a missing/revoked key does not stop the server. `earth_engine_status()` feeds `/api/config`, and cloud entry points (`/api/search-images`, `/api/search-s1-images`, `/api/process-image`, `/api/process-s1-image`) call `require_earth_engine()` for a 503 with the reason. Call the guard *before* the handler's `try:` so it is not swallowed into a 500. This is deliberately cloud-path-only — the local path must keep working without GEE.
- **Parallel downloads:** Mosaic builder uses 6 parallel workers for tile downloads from GEE
- **In-memory caching:** Raster data, spectral indices, and visualization results are cached

### Dual-path architecture (cloud vs local) — ALWAYS EDIT BOTH

Every analysis feature — **target detection, spectral analysis, SAM3, change detection, Time A / Time B** — has two parallel implementations that must stay in lockstep:

- **Cloud (satellite search / GEE) path:** `backend/api/routes_target_detection.py`, `routes_mangrove_segmentation.py`, `routes_sam3.py`, `routes_change_detection.py`. Caches: `TARGET_DETECTION_CACHE`, `MANGROVE_SEG_CACHE`, `SAM3_MASK_CACHE`, `CHANGE_DETECTION_CACHE`. Results carry `transform` / `crs` / `bbox`; overlays are warped to AOI via `warp_rgb_and_mask_to_aoi(scale_m=10)` and placed on the Leaflet map.
- **Local / uploaded image path:** `backend/api/routes_local.py` (`/api/local/...` and `/api/local/uploaded/...` endpoints). Caches: `_LOCAL_TD_CACHE`, `_LOCAL_INDEX_CACHE`, `_LOCAL_RASTER_CACHE`; the SAM3 cache is shared with the cloud path (`SAM3_MASK_CACHE`, with `transform`/`crs` set to `None` for uploaded entries). Results have no CRS/transform; overlays are encoded at native pixel resolution.
- **Frontend controllers** branch on `isLocalMode()` and route overlays through either `mapManager.showAnalysisLayer` (cloud, geo bounds) or `localImage.showLocalAnalysisLayer` (local, pixel size).

**Rule:** Any fix, refactor, or new feature touching one path **must be applied to the other at the same time**. When grep-ing to scope a change, always search both `routes_*.py` and `routes_local.py`, and both the cloud cache names and the `_LOCAL_*_CACHE` names. Report completion as "cloud + local both updated" (or explicitly flag when a change legitimately applies to only one side, e.g. bbox handling).

## Notes

- No test suite exists in this project
- Documentation (`docs/FUNCTIONAL_SPECIFICATION.md`) is in Korean
- CORS is open (`allow_origins=["*"]`)
- The `ee-service-account-key.json` file is required but git-ignored
