"""
API routes module.
"""

from .routes_search import router as search_router
from .routes_process import router as process_router
from .routes_analysis import router as analysis_router
from .routes_download import router as download_router
from .schemas import *

