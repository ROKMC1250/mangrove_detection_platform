# EarthScope

**A web-based Earth-observation image analysis platform.** Search, visualize and analyze Sentinel-2 optical and Sentinel-1 SAR imagery — or your own GeoTIFFs — from a single map-based UI: spectral indices, hyperspectral target detection, deep-learning segmentation, and two-date change detection.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Leaflet](https://img.shields.io/badge/Leaflet-199900?logo=leaflet&logoColor=white)
![Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?logo=google&logoColor=white)

EarthScope runs entirely on your own machine and your own Google Earth Engine credentials — there is no hosted service. Clone it, put your service account key in place, edit one `.env` file, and run it.

---

## Demo

<video src="https://github.com/ROKMC1250/mangrove_detection_platform/releases/download/v2.0/EarthScope_2.mp4" controls muted width="100%"></video>

▶ **[Watch the walkthrough](https://github.com/ROKMC1250/mangrove_detection_platform/releases/download/v2.0/EarthScope_2.mp4)** (if the player above does not load in your browser)

---

## Highlights

| | |
|---|---|
| **One continuous workflow** | Search → process → analyze → tune thresholds → compare dates, all on one screen |
| **Cloud *and* local imagery** | Earth Engine scenes and your own uploaded GeoTIFFs run through the same analysis tools |
| **Classical *and* deep learning** | Spectral detectors (SAM/ACE/RXD/CEM) alongside Segformer, UNet++ and SAM3 |
| **Two-date comparison** | Time A / Time B slots keep two independent analyses side by side for change detection |
| **Interactive exploration** | Click the map to read pixel spectra; re-threshold results without recomputing |

---

## Features

### Image search & processing
- **Scene search** — draw an AOI (rectangle, SHP, KML or coordinates), pick a date range and a cloud-cover ceiling, and browse matching Sentinel-2 / Sentinel-1 scenes with thumbnails, acquisition dates and cloud percentages.
- **Processing** — selected scenes are downloaded from Earth Engine as parallel tiles (6 workers), mosaicked, cached to disk and loaded onto the GPU. A cloud-mask overlay is generated from the Sentinel-2 SCL band.
- **Export** — download the processed scene as GeoTIFF (Sentinel-2: 13 bands including SCL; Sentinel-1: VV + VH) or export the current view as a rendered PNG.
- **Symbology** — drag-and-drop RGB band assignment, min/max and cumulative-percentage stretch, gamma / contrast / brightness / saturation controls, and keyboard shortcuts (`1`–`9`) to flip between band combinations.

### Spectral analysis
- Built-in indices: **NDVI, NDMI, MVI, NDWI, SAVI, EVI**, plus arbitrary custom band formulas.
- Apply a threshold range to any index to produce a binary mask (detections drawn in green).
- Click any pixel for its index value and full per-band spectrum; select a region for spectral statistics (mean, standard deviation).

### Target detection
- Algorithms: **SAM** (Spectral Angle Mapper), **ACE** (Adaptive Cosine Estimator), **RXD** (Reed–Xiaoli), **CEM** (Constrained Energy Minimization), and the neural **MLP_AMF** / **MLP_ACE** detectors.
- Mark positive and negative training points directly on the map. MLP variants stream training progress (step / loss) over Server-Sent Events.
- Inspect score distributions and spectral comparison charts, then re-apply thresholds without rerunning the detector.

### Segmentation
- **SAM3** — point/box prompting (single best mask) and text prompting (open-vocabulary PCS mode returning multiple instances). Image embeddings are encoded once and cached.
- **Mangrove segmentation** — Segformer (`mit_b2`) or UNet++ over 13 Sentinel-2 channels, patch-based inference (256 px, 50 % overlap, Gaussian blending) with optional test-time augmentation.
- **Flood segmentation** — UNet++ (`resnet34`) on Sentinel-1 VV, producing a probability map you threshold into a water mask, plus an eraser tool that drops false-positive blobs by connected-component labelling.

### Change detection
- Compare the binary masks held in the Time A and Time B slots.
- **Gained** (green) and **lost** (red) overlays with pixel counts and area ratios.
- Compose several masks into one input using include / exclude / add operators.

### Interactive tools
- **Spectral Inspector** — accumulate spectral profiles by clicking the map and compare them on a multi-series line chart.
- **Dual-map compare** — split view with synchronized pan/zoom and a draggable divider.
- **Layer management** — toggle visibility, reorder by drag, rename or delete layers, and combine masks by intersection / union / exclusion.

---

## Supported data

| Source | Details |
|---|---|
| **Sentinel-2 SR Harmonized** | 12 spectral bands + SCL classification band. B2/B3/B4/B8 native 10 m; B5–B7/B8A/B11/B12 20 m; B1/B9 60 m — all bicubic-resampled to 10 m at download. Filtered by `CLOUDY_PIXEL_PERCENTAGE`. Target detection uses the 10 non-atmospheric bands (B1 and B9 excluded). |
| **Sentinel-1 GRD** | VV / VH polarizations, IW mode, ascending and descending orbits, dB-scaled to [-30, 10]. Used for flood segmentation. |
| **Uploaded GeoTIFF** | Upload a multi-band GeoTIFF and assign band roles (Blue / Green / Red / NIR / SWIR …) manually, or browse band-registration experiment outputs from a local directory. |
| **AlphaEarth Satellite Embeddings** *(optional)* | V1 annual collection, bands A01 / A16 / A32 at 10 m, for auxiliary visualization. |

---

## Architecture

```
[Leaflet map UI] ──(/api/*)──> [FastAPI backend] ──> [Google Earth Engine]
  │ PlatformController            │ single-session gate      │ Sentinel-2 / Sentinel-1
  │ Time A / Time B slots         │ job-id progress tracking └─> parallel tile download (6 workers)
  │ isLocalMode() branching       │ GPU raster cache (torch)
  └ map / analysis / inspector    └─> PyTorch models (Segformer · UNet++ · SAM3)
```

**Backend** — FastAPI, entry point `backend/main.py`.

- `backend/api/` — routes split by domain: `search`, `process`, `analysis`, `download`, `target_detection`, `sam3`, `mangrove_segmentation`, `flood_segmentation`, `change_detection`, `session`, and `local` (uploaded / on-disk imagery).
- `backend/services/` — `earth_engine`, `downloader`, `spectral_analysis`, `target_detection`, `model_inference`, `flood_inference`, `sam3_service`, `change_detection`, `gpu_compute`, `visualization`.
- `backend/core/` — `config` (paths, band definitions, credentials, HTTP pooling), `progress` (phase-weighted job progress), `session_gate`.

**Frontend** — vanilla JavaScript + Leaflet, no build step, no bundler.

- `frontend/index.html` — application shell; `frontend/script.js` — `PlatformController`, event orchestration, slots, tabs, shortcuts.
- `frontend/js/` — `api-client`, `map-core` / `map-drawing` / `map-layers` / `map-file-loader`, `image-search`, `image-processor`, and the `analysis`, `threshold`, `target-detection`, `sam3`, `spectral-inspector`, `local-image`, `dual-map` and `change-detection` controllers.

### Cloud and local paths

Every analysis feature exists twice — once for Earth Engine scenes and once for local imagery — and the two are kept in lockstep. Worth knowing before you extend the code:

| | Cloud (Earth Engine) | Local (uploaded / on-disk) |
|---|---|---|
| Data | Sentinel-2 / Sentinel-1 via GEE | Uploaded GeoTIFF, `LOCAL_BASE_DIR` results |
| Endpoints | `/api/...` | `/api/local/...` |
| Georeferencing | `transform` / `crs` / `bbox`, warped to the AOI | none — overlays encoded at native pixel resolution |
| Result caches | `TARGET_DETECTION_CACHE`, `MANGROVE_SEG_CACHE`, `CHANGE_DETECTION_CACHE` | `_LOCAL_TD_CACHE`, `_LOCAL_INDEX_CACHE`, `_LOCAL_RASTER_CACHE` (SAM3 cache shared) |

---

## Getting started

### Prerequisites

- **Python 3.12+** (required by SAM3)
- **[uv](https://docs.astral.sh/uv/)** — installed automatically by `run.sh` if missing
- **Git LFS** — the segmentation checkpoints are stored as LFS objects
- **NVIDIA GPU with CUDA 12.6+** — strongly recommended; PyTorch is installed from the cu128 wheel index. CPU-only works but inference is slow.
- **A Google Cloud project registered for Earth Engine**, with a service account key (see [Configuration](#configuration))

### Install and run

```bash
git clone https://github.com/ROKMC1250/mangrove_detection_platform.git
cd mangrove_detection_platform

# fetch the model checkpoints stored in Git LFS
git lfs install
git lfs pull

# place your Earth Engine service account key
cp /path/to/your-key.json backend/ee-service-account-key.json

# configure — this is the only file you need to edit
cp .env.example .env
$EDITOR .env

bash run.sh          # Windows: run.bat
```

`run.sh` creates the virtual environment, installs PyTorch and the remaining dependencies, installs SAM3 from source if needed, and starts the server:

```
http://localhost:8000
```

The frontend is served as static files from `/static/`; the API lives under `/api/`.

To start the server manually once the environment exists:

```bash
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Configuration

Everything deployment-specific lives in **one file: `.env`** at the project root. Copy `.env.example` to `.env` and edit these values — nothing else in the codebase needs to change.

| Variable | Required | What it is |
|---|---|---|
| `EE_SERVICE_ACCOUNT_KEY` | **yes** | Path to your Earth Engine service account JSON. Relative paths resolve against the project root. Default: `backend/ee-service-account-key.json` |
| `EE_SERVICE_ACCOUNT` | no | Service account address. Leave empty and it is read from `client_email` in the key file |
| `GCS_BUCKET` | no | Cloud Storage bucket for server-side GeoTIFF export of large AOIs. Empty disables that export path; everything else still works |
| `GCS_BUCKET_REGION` | no | Region of that bucket (e.g. `asia-northeast3`) |
| `SAM3_CHECKPOINT_DIR` | no | Directory holding the SAM3 checkpoints. Default: `repo/sam3` |
| `SAM3_BPE_PATH` | no | Explicit path to the SAM3 BPE vocabulary. Empty lets the `sam3` package resolve its own bundled asset |
| `MODEL_ROOT` | no | Root of the segmentation model source tree. Default: `repo/mangrove_segmentation` |
| `LOCAL_BASE_DIR` | no | Directory scanned in local mode for band-registration experiment results. Default: `experiment_results` |
| `HTTP_POOL` | no | HTTP connection pool size for Earth Engine downloads (default `32`) |
| `PARALLEL_TILE_WORKERS` | no | Parallel tile download workers (default `6`) |

> `.env` and `backend/ee-service-account-key.json` are both git-ignored. Keep them that way — a committed service account key gives anyone read/write access to your Earth Engine and Cloud Storage quota.

If the key is missing or rejected the server still starts: `/api/config` reports `earth_engine.ready = false` with the reason, Earth Engine endpoints answer `503` explaining how to fix it, and uploaded-image (local) analysis keeps working. That makes it possible to explore the UI before wiring up Google Cloud.

### Setting up Earth Engine access

1. Create (or pick) a Google Cloud project and **enable the Earth Engine API**.
2. **Register the project** for Earth Engine use at [code.earthengine.google.com/register](https://code.earthengine.google.com/register).
3. Create a **service account** in that project (IAM & Admin → Service Accounts).
4. Grant it Earth Engine access — the *Earth Engine Resource Viewer* role is enough for reading imagery. Add *Storage Object Admin* on your bucket only if you want the GCS export path.
5. Create a **JSON key** for the service account and save it as `backend/ee-service-account-key.json` (or point `EE_SERVICE_ACCOUNT_KEY` somewhere else).

```bash
gcloud iam service-accounts create earthscope \
  --display-name="EarthScope service account"

gcloud iam service-accounts keys create backend/ee-service-account-key.json \
  --iam-account=earthscope@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### Tuning the models

Model behaviour is configured in **`backend/model_config.yaml`**:

| Key | Meaning |
|---|---|
| `model_dir` | Directory of the trained mangrove model. Empty → `repo/mangrove_segmentation/checkpoints/segformer_MVI_v1` |
| `checkpoint` | Weight file to load (`last.pt` or `best.pt`) |
| `gpus` | `"0"` for the first GPU, empty for CPU |
| `patch_size`, `overlap` | Patch-based inference geometry (256 px, 0.5 = 50 % overlap) |
| `use_tta` | Test-time augmentation — more accurate, roughly 4× slower |
| `default_model` | Architecture used when a checkpoint ships no `config.yaml` (`name`, `encoder_name`, `in_channels`, `classes`) |
| `flood_*` | Flood model equivalents: directory, checkpoint, 512 px patches, 128 px overlap, batch size, blend mode, SAR dB range |

---

## Model weights

Weights are loaded from `<model_dir>/weights/<checkpoint>`:

| Model | Expected location | Bundled? |
|---|---|---|
| Mangrove Segformer (`mit_b2`, 13ch) | `repo/mangrove_segmentation/checkpoints/segformer_MVI_v1/weights/{last,best}.pt` | ✅ via Git LFS |
| SAM2 (legacy target-detection path) | `repo/sam2/sam2_hiera_small.pt` | ✅ via Git LFS |
| Flood UNet++ (`resnet34`, S1 VV) | `repo/flood_segmentation/checkpoints/unetpp_vh_v1/weights/best.pt` | ❌ bring your own — the matching `config.yaml` is included |
| SAM3 | `$SAM3_CHECKPOINT_DIR` (default `repo/sam3`) | ❌ pulled from Hugging Face |

SAM3 needs a Hugging Face login and an install from source — `run.sh` handles the install:

```bash
hf auth login
pip install git+https://github.com/facebookresearch/sam3.git
```

**Missing models degrade gracefully.** If a checkpoint is absent, that one feature reports itself unavailable via `/api/model-status` while search, visualization, spectral analysis and classical target detection keep working.

---

## Typical workflows

1. **Optical analysis** — draw an AOI → search Sentinel-2 → select and visualize a scene → run a spectral index, target detector or SAM3 → tune the threshold → export the mask.
2. **SAR flood mapping** — draw an AOI → search Sentinel-1 → select a scene → run flood segmentation → threshold the probability map → erase false-positive blobs.
3. **Local imagery** — upload a GeoTIFF → assign band roles → visualize → run target detection, SAM3 or a custom index formula.
4. **Change detection** — analyze Time A → switch to the Time B slot → analyze a second date → open the change-detection tab for gained/lost statistics.
5. **Side-by-side comparison** — *Compare Layers* → assign a layer to each pane → inspect with synchronized pan and zoom.

---

## API surface

| Endpoint | Purpose |
|---|---|
| `POST /api/search-images`, `/api/search-s1-images` | Scene search for Sentinel-2 / Sentinel-1 |
| `POST /api/process-image`, `/api/process-s1-image` | Download, mosaic and cache a scene |
| `GET /api/progress?job_id=…` | Phase-weighted progress for long-running jobs |
| `POST /api/compute-spectral-index`, `/api/apply-threshold-range` | Spectral indices and masking |
| `GET /api/available-indices` | Index catalogue |
| `POST /api/get-pixel-value`, `/api/get-spectral-values`, `/api/get-area-spectral-stats` | Pixel and region spectra |
| `POST /api/target-detection/run`, `/run-stream`, `/apply-threshold` | Target detection (streaming variant for MLP training) |
| `POST /api/sam3/encode`, `/predict`, `/text-predict`, `/save-mask` | SAM3 segmentation |
| `POST /api/mangrove-segmentation/run`, `/apply-threshold` | Mangrove segmentation |
| `POST /api/flood-segmentation/run`, `/apply-threshold`, `/erase-region`, `/reset-exclusions` | Flood segmentation |
| `POST /api/change-detection/run` | Time A vs Time B comparison |
| `POST /api/download-s2-image`, `/api/download-s1-image` | GeoTIFF export |
| `GET /api/proxy-file?path=…` | Serve results from `outputs/` (supports HTTP Range) |
| `POST /api/local/...` (31 endpoints) | Local-mode mirrors of the analysis features |
| `POST /api/session/acquire`, `/heartbeat`, `/release` · `GET /api/session/status` | Single-session gate |
| `GET /api/config`, `/api/model-status`, `/health` | Runtime configuration and health |

---

## Project layout

```
.
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── api/                    # route handlers per domain (+ routes_local.py)
│   ├── services/               # Earth Engine, download, analysis, inference
│   ├── core/                   # config, progress tracking, session gate
│   ├── utils/                  # caching, IO safety
│   └── model_config.yaml       # model selection & inference parameters
├── frontend/
│   ├── index.html              # single-page shell
│   ├── script.js               # PlatformController
│   ├── js/                     # map, search, analysis, detection controllers
│   └── assets/                 # fonts and logo
├── repo/                       # model source trees and checkpoints
│   ├── mangrove_segmentation/
│   ├── flood_segmentation/
│   ├── Target_detection/
│   └── sam2/
├── outputs/                    # generated rasters, masks, exports (git-ignored)
├── .env.example                # ← copy to .env and edit
├── run.sh / run.bat            # environment setup + server start
└── requirements: backend/requirements.txt
```

---

## Acknowledgements

- [Google Earth Engine](https://earthengine.google.com/) — imagery catalogue and processing
- [SAM 3](https://github.com/facebookresearch/sam3) (Meta AI) — promptable and open-vocabulary segmentation
- [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) — Segformer / UNet++ implementations
- [Leaflet](https://leafletjs.com/) — map rendering · [Chart.js](https://www.chartjs.org/) — spectral charts
- [Pretendard](https://github.com/orioncactus/pretendard) — UI typeface
