# EarthScope

**A web-based satellite image analysis platform for Earth observation.** Search Sentinel-2 optical and Sentinel-1 SAR imagery, or upload your own GeoTIFF, and analyze it on a single map: spectral indices, target detection, deep-learning segmentation and two-date change detection.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA%20accelerated-76B900?logo=nvidia&logoColor=white)
![Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?logo=google&logoColor=white)
![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-lightgrey)

Everything runs on your own machine with your own Google Earth Engine credentials — there is no hosted service. **The whole pipeline is GPU-accelerated**: rasters stay resident on the GPU between steps, indices and detectors are computed as batched tensor operations, and segmentation runs as batched patch inference, so re-thresholding or switching band combinations is interactive instead of a re-run. Without a GPU it falls back to CPU and simply runs slower.

---

## Demo

![EarthScope walkthrough](.github/media/earthscope-demo.gif)

<sub>Complete walkthrough, 3× speed. [Original 1080p recording ↗](https://github.com/ROKMC1250/Earthscope/releases/download/v2.0/EarthScope_2.mp4)</sub>

---

## What it can do

**Search & visualize** — draw an area of interest (rectangle, SHP, KML or coordinates), filter by date range and cloud cover, and browse matching scenes. Assign RGB bands by drag-and-drop, stretch with min/max or cumulative percentages, and adjust gamma, contrast, brightness and saturation. Export as GeoTIFF or PNG.

**Spectral analysis** — NDVI, NDMI, MVI, NDWI, SAVI, EVI or your own band formula. Apply a threshold range to turn any index into a mask, click any pixel to read its full spectrum, or select a region for spectral statistics.

**Target detection** — mark positive and negative examples on the map and run SAM (Spectral Angle Mapper), ACE, RXD, CEM or the neural MLP detectors. Score distributions and spectral comparisons are shown as charts, and thresholds can be re-applied without recomputing.

**Segmentation** — SAM3 with point, box or open-vocabulary text prompts; a Segformer / UNet++ mangrove model over 13 Sentinel-2 bands; and a UNet++ flood model on Sentinel-1 VV with an eraser for false-positive water blobs.

**Change detection** — analyze two dates in the Time A / Time B slots and compare them: gained and lost areas with pixel counts and ratios.

**Comparison tools** — accumulate spectral profiles by clicking the map, compare two layers side by side with synchronized pan and zoom, and combine masks by intersection, union or exclusion.

### Data it works with

| Source | What you get |
|---|---|
| **Sentinel-2 SR Harmonized** | 12 spectral bands + scene classification, all resampled to 10 m, cloud-cover filtered |
| **Sentinel-1 GRD** | VV / VH polarizations, IW mode, ascending and descending orbits |
| **Your own GeoTIFF** | Upload a multi-band file and assign band roles manually |

---

## Getting started

**Requirements:** Python 3.12+, Git LFS, and an NVIDIA GPU with CUDA 12.6+ (recommended, not required).

### 1. Get the code

```bash
git clone https://github.com/ROKMC1250/Earthscope.git
cd Earthscope
git lfs install && git lfs pull      # fetches the segmentation checkpoints
```

### 2. Get an Earth Engine key

Satellite search needs a Google Earth Engine service account:

1. Create a Google Cloud project and enable the **Earth Engine API**.
2. Register the project at **[code.earthengine.google.com/register](https://code.earthengine.google.com/register)**.
3. Create a service account in that project and grant it *Earth Engine Resource Viewer*.
4. Download a **JSON key** and save it as `backend/ee-service-account-key.json`.

```bash
gcloud iam service-accounts keys create backend/ee-service-account-key.json \
  --iam-account=YOUR_ACCOUNT@YOUR_PROJECT.iam.gserviceaccount.com
```

### 3. Get a Hugging Face token (only for SAM3)

SAM3 weights are gated. Request access at **[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)**, then:

```bash
hf auth login
```

Everything except SAM3 works without this.

### 4. Configure and run

```bash
cp .env.example .env     # then edit .env — this is the only file you need to change
bash run.sh              # Windows: run.bat
```

Open **http://localhost:8000**. `run.sh` creates the virtual environment, installs the dependencies and starts the server.

---

## Configuration

All settings live in **`.env`** at the project root — copy `.env.example` and edit it. Nothing else needs changing.

| Variable | Required | What it is |
|---|---|---|
| `EE_SERVICE_ACCOUNT_KEY` | **yes** | Path to your Earth Engine JSON key |
| `GCS_BUCKET` / `GCS_BUCKET_REGION` | no | Cloud Storage bucket, for exporting very large areas |
| `SAM3_CHECKPOINT_DIR` | no | Where the SAM3 weights live |
| `LOCAL_BASE_DIR` | no | Folder scanned when browsing your own images |

Model choice, patch size, overlap and test-time augmentation are set in `backend/model_config.yaml`.

> Keep `.env` and your JSON key out of version control — both are git-ignored by default. A leaked service account key gives anyone access to your Earth Engine and Cloud Storage quota.

If the key is missing or rejected the server still starts: satellite search returns a clear error explaining what to fix, and analysis of uploaded images keeps working. The mangrove, flood and SAM3 models are each optional too — whatever is unavailable simply reports itself as such.

---

## License

**[PolyForm Noncommercial License 1.0.0](LICENSE) — noncommercial use only.**

You may use, modify and redistribute EarthScope for any noncommercial purpose: personal projects, study and research, and use by educational institutions, public research bodies, government institutions and nonprofit organizations. **Commercial use is not permitted** — for a commercial licence, contact the copyright holder.

Third-party code, model weights and fonts included in this repository remain under their own licenses.

## Acknowledgements

[Google Earth Engine](https://earthengine.google.com/) · [SAM 3](https://github.com/facebookresearch/sam3) (Meta AI) · [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) · [Leaflet](https://leafletjs.com/) · [Chart.js](https://www.chartjs.org/) · [Pretendard](https://github.com/orioncactus/pretendard)
