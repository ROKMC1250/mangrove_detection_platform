"""
EarthScope Backend - Main Application Entry Point

This is the modular version of the backend that organizes code into:
- core/: Configuration and progress tracking
- services/: Business logic (Earth Engine, downloads, analysis, etc.)
- api/: FastAPI route handlers
- utils/: Utility functions (caching, etc.)
"""

import os
import mimetypes
import time as _time
from datetime import datetime as _dt

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

# Initialize core components
from .core.config import (
    PROJECT_ROOT,
    FRONTEND_DIR,
    OUTPUTS_DIR,
    STATIC_MOUNT,
    GCS_BUCKET,
)
from .core.progress import PROGRESS_TRACKER

# Initialize Earth Engine
from .services.earth_engine import init_earth_engine, earth_engine_status
from .services.model_inference import init_model1, get_model1_status
from .services.sam3_service import init_sam3, get_sam3_status
from .services.flood_inference import init_flood_model, get_flood_model_status
from .services.session_gate import SESSION_GATE

# Import API routers
from .api import (
    search_router,
    process_router,
    analysis_router,
    download_router,
)
from .api.routes_target_detection import router as target_detection_router
from .api.routes_mangrove_segmentation import router as mangrove_segmentation_router
from .api.routes_flood_segmentation import router as flood_segmentation_router
from .api.routes_change_detection import router as change_detection_router
from .api.routes_local import router as local_router
from .api.routes_sam3 import router as sam3_router
from .api.routes_session import router as session_router, SESSION_COOKIE


# Initialize Earth Engine at module load. A credential failure is non-fatal:
# the server still starts and local (uploaded-image) analysis keeps working,
# while GEE-backed endpoints answer 503 with the reason.
init_earth_engine()


# Create FastAPI application
app = FastAPI(
    title="EarthScope Backend",
    description="Backend API for satellite image processing and Earth observation analysis",
    version="2.0.0"
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Session gate — admit only one browser session at a time.
# Everything outside _OPEN_PATHS is blocked for callers who don't hold the
# gate. The frontend shows a full-screen waiting overlay on 423.
_OPEN_PATHS = (
    "/static/", "/outputs/", "/api/session/", "/health",
)


@app.middleware("http")
async def session_gate_middleware(request: Request, call_next):
    path = request.url.path
    if path in ("", "/") or any(path.startswith(p) for p in _OPEN_PATHS):
        return await call_next(request)

    sid = request.cookies.get(SESSION_COOKIE)
    if not SESSION_GATE.holds(sid):
        return JSONResponse(
            status_code=423,
            content={
                "detail": "Another user is currently using the platform. Please wait.",
                "gate": SESSION_GATE.status(),
            },
        )
    return await call_next(request)


# In-memory set of session cookie values seen since process start. Used
# only to flag the first request that introduces a new sid so the operator
# can tell at a glance when a new computer/browser starts hitting the
# server. The session gate itself only ever admits one sid at a time, so
# this set stays tiny in practice.
_SEEN_SIDS: set = set()

# Static / output traffic fires on every page load and is rarely useful
# for "who's using the platform" diagnostics. Skip these paths to keep
# the access log readable.
_LOG_SKIP_PREFIXES = ("/static/", "/outputs/")


@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    """Single-line access log with timestamp, client IP, and short session
    id. Sits outside `session_gate_middleware` on purpose so 423 (locked
    out) responses are also recorded — those are the ones that tell you a
    second computer is trying to connect."""
    start = _time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (_time.perf_counter() - start) * 1000

    path = request.url.path
    if any(path.startswith(p) for p in _LOG_SKIP_PREFIXES):
        return response

    # Prefer X-Forwarded-For when a reverse proxy is in front of uvicorn,
    # otherwise fall back to the direct peer address.
    fwd = request.headers.get("x-forwarded-for")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "-")

    sid_full = request.cookies.get(SESSION_COOKIE) or ""
    sid_short = sid_full[:8] if sid_full else "-"
    is_new = bool(sid_full) and sid_full not in _SEEN_SIDS
    if is_new:
        _SEEN_SIDS.add(sid_full)

    ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    new_tag = " [NEW]" if is_new else ""
    print(
        f"[{ts}] {ip:<15} sid={sid_short}{new_tag} "
        f"{request.method:<6} {path} {response.status_code} ({elapsed_ms:.0f}ms)",
        flush=True,
    )
    return response


# Mount static directories
app.mount(STATIC_MOUNT, StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


# Include API routers
app.include_router(session_router)
app.include_router(search_router)
app.include_router(process_router)
app.include_router(analysis_router)
app.include_router(download_router)
app.include_router(target_detection_router)
app.include_router(mangrove_segmentation_router)
app.include_router(flood_segmentation_router)
app.include_router(change_detection_router)
app.include_router(local_router)
app.include_router(sam3_router)


# ===== Core Endpoints =====

@app.get("/api/config")
async def get_config():
    """Return configuration information."""
    return {
        "use_gee": True,
        "use_copernicus": True,
        "use_public_stac": False,
        "gcs_bucket_configured": bool(GCS_BUCKET),
        "earth_engine": earth_engine_status(),
    }


@app.get("/api/model-status")
async def model_status():
    """Get the status of the segmentation model."""
    return get_model1_status()


@app.get("/api/progress")
async def get_progress(job_id: str):
    """Get progress status for a job."""
    status = PROGRESS_TRACKER.get_status(job_id)
    if not status:
        print(f"PROGRESS API - No data found for job_id: {job_id}, returning pending")
        return {
            "job_id": job_id,
            "percent": 0,
            "message": "Extracting analysis results...",
            "status": "pending",
            "current_phase": None,
            "estimated_remaining_seconds": None,
            "phases": []
        }
    print(f"PROGRESS API - Returning data for job_id: {job_id} -> {status['percent']}% ({status['status']})")
    return status


@app.get("/api/proxy-file")
async def proxy_file(path: str, request: Request):
    """Stream a local file to the client with Range Request support for COG files."""
    try:
        if path.startswith("file://"):
            path = path[7:]
        abs_path = os.path.abspath(path)
        
        # Security check - only allow files from outputs directory
        if not abs_path.startswith(OUTPUTS_DIR):
            raise HTTPException(status_code=400, detail="Invalid file path")
        if not os.path.exists(abs_path):
            raise HTTPException(status_code=404, detail="File not found")

        file_size = os.path.getsize(abs_path)
        mime, _ = mimetypes.guess_type(abs_path)
        media_type = mime or "application/octet-stream"

        # Handle Range requests for COG support
        range_header = request.headers.get("range")
        if range_header:
            # Parse range header: "bytes=start-end"
            range_match = range_header.replace("bytes=", "").split("-")
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if range_match[1] else file_size - 1
            
            # Ensure valid range
            start = max(0, min(start, file_size - 1))
            end = max(start, min(end, file_size - 1))
            content_length = end - start + 1
            
            def iterfile_range():
                with open(abs_path, "rb") as f:
                    f.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk_size = min(1024 * 1024, remaining)
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            }
            return StreamingResponse(
                iterfile_range(),
                status_code=206,
                media_type=media_type,
                headers=headers
            )

        # No range request - return full file
        def iterfile():
            with open(abs_path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        }
        return StreamingResponse(iterfile(), media_type=media_type, headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Serve the main index.html."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=500, detail="index.html not found")
    return FileResponse(index_path)


# ===== Startup Event =====

@app.on_event("startup")
async def on_startup():
    """Initialize components on startup."""
    print("🚀 Starting EarthScope Backend...")
    try:
        model_loaded = init_model1()
        if model_loaded:
            print("✅ Segmentation model loaded successfully")
        else:
            print("⚠️  Segmentation model not available - continuing without it")
            print("   Other analysis features (NDVI, NDMI, MVI, etc.) are still available")
    except Exception as e:
        print(f"⚠️  STARTUP - Model initialization error: {e}")
        print("   Continuing without segmentation model - other features are still available")

    # Initialize SAM3 model
    try:
        sam3_loaded = init_sam3()
        if sam3_loaded:
            print("✅ SAM3 model loaded successfully")
        else:
            print("⚠️  SAM3 model not available - continuing without it")
    except Exception as e:
        print(f"⚠️  STARTUP - SAM3 initialization error: {e}")
        print("   Continuing without SAM3 - other features are still available")

    # Initialize flood segmentation model (UNet++ on Sentinel-1 VV)
    try:
        flood_loaded = init_flood_model()
        if flood_loaded:
            print("✅ Flood segmentation model loaded successfully")
        else:
            print("⚠️  Flood segmentation model not available - continuing without it")
    except Exception as e:
        print(f"⚠️  STARTUP - Flood model initialization error: {e}")
        print("   Continuing without flood segmentation - other features are still available")



# ===== Health Check =====

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}

