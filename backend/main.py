"""
Mangrove Platform Backend - Main Application Entry Point

This is the modular version of the backend that organizes code into:
- core/: Configuration and progress tracking
- services/: Business logic (Earth Engine, downloads, analysis, etc.)
- api/: FastAPI route handlers
- utils/: Utility functions (caching, etc.)
"""

import os
import mimetypes

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

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
from .services.earth_engine import init_earth_engine
from .services.model_inference import init_model1, get_model1_status

# Import API routers
from .api import (
    search_router,
    process_router,
    analysis_router,
    download_router,
)
from .api.routes_target_detection import router as target_detection_router


# Initialize Earth Engine at module load
init_earth_engine()


# Create FastAPI application
app = FastAPI(
    title="Mangrove Platform Backend",
    description="Backend API for satellite image processing and mangrove analysis",
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


# Mount static directories
app.mount(STATIC_MOUNT, StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


# Include API routers
app.include_router(search_router)
app.include_router(process_router)
app.include_router(analysis_router)
app.include_router(download_router)
app.include_router(target_detection_router)


# ===== Core Endpoints =====

@app.get("/api/config")
async def get_config():
    """Return configuration information."""
    return {
        "use_gee": True,
        "use_copernicus": True,
        "use_public_stac": False,
        "gcs_bucket_configured": bool(GCS_BUCKET),
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
    print("🚀 Starting Mangrove Platform Backend...")
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


# ===== Health Check =====

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}

