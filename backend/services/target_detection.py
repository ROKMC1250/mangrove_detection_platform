"""
Target Detection Service Module

Provides various spectral target detection algorithms for satellite imagery.
Supports SAM (Spectral Angle Mapper), ACE (Adaptive Cosine Estimator),
RXD (Reed-Xiaoli Detector), and CEM (Constrained Energy Minimization).
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import rasterio


# =============================================================================
# Sentinel-2 band policy for target detection
# =============================================================================
# Hardcoded: B1 (60m Coastal aerosol) and B9 (60m Water vapor) are always excluded
# from target detection — they are atmospheric-correction bands, low resolution,
# and hurt detector performance on surface targets.
# Raster band order from process-image (12 bands): B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12
S2_TD_KEEP_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]  # B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
S2_TD_BAND_NAMES = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']


def drop_s2_60m_bands(loader: 'ImageDataLoader') -> None:
    """Drop B1 and B9 from a 12-band Sentinel-2 raster loader in place."""
    loader.data = loader.data[:, :, S2_TD_KEEP_INDICES]


# =============================================================================
# Base Detector
# =============================================================================

class BaseDetector(ABC):
    """Abstract base class for target detectors."""
    
    def __init__(self, name: str = "BaseDetector"):
        self.name = name
        self._background_mean: Optional[np.ndarray] = None
        self._background_cov: Optional[np.ndarray] = None
        self._background_cov_inv: Optional[np.ndarray] = None
        self._is_fitted = False
    
    def fit_background(self, background_spectra: np.ndarray) -> 'BaseDetector':
        """Compute background statistics."""
        self._background_mean = np.mean(background_spectra, axis=0)
        self._background_cov = np.cov(background_spectra, rowvar=False)
        
        # Handle 1D case
        if self._background_cov.ndim == 0:
            self._background_cov = np.array([[self._background_cov]])
        
        # Regularize for numerical stability
        if self._background_cov.shape[0] > 0:
            reg = 1e-6 * np.trace(self._background_cov) / self._background_cov.shape[0]
            self._background_cov += reg * np.eye(self._background_cov.shape[0])
        
        try:
            self._background_cov_inv = np.linalg.inv(self._background_cov)
        except np.linalg.LinAlgError:
            self._background_cov_inv = np.linalg.pinv(self._background_cov)
        
        self._is_fitted = True
        return self
    
    @abstractmethod
    def detect(self, image_data: np.ndarray, target_spectrum: np.ndarray) -> np.ndarray:
        """Perform detection. Returns 2D detection map."""
        pass
    
    def _higher_is_target(self) -> bool:
        """Whether higher values indicate target."""
        return True
    
    @property
    def description(self) -> str:
        """Get algorithm description."""
        return "Base detector"


# =============================================================================
# SAM Detector
# =============================================================================

class SAMDetector(BaseDetector):
    """Spectral Angle Mapper - measures angle between spectra."""
    
    def __init__(self):
        super().__init__(name="SAM")
    
    def detect(self, image_data: np.ndarray, target_spectrum: np.ndarray) -> np.ndarray:
        height, width, n_bands = image_data.shape
        pixels = image_data.reshape(-1, n_bands)
        
        target_norm = np.linalg.norm(target_spectrum)
        pixel_norms = np.linalg.norm(pixels, axis=1)
        
        pixel_norms = np.maximum(pixel_norms, 1e-10)
        target_norm = max(target_norm, 1e-10)
        
        dot_products = np.dot(pixels, target_spectrum)
        cos_angles = np.clip(dot_products / (pixel_norms * target_norm), -1.0, 1.0)
        angles = np.arccos(cos_angles)
        
        return angles.reshape(height, width)
    
    def _higher_is_target(self) -> bool:
        return False  # Lower angle = more similar
    
    @property
    def description(self) -> str:
        return "Spectral Angle Mapper - measures spectral similarity by angle"


# =============================================================================
# ACE Detector
# =============================================================================

class ACEDetector(BaseDetector):
    """Adaptive Cosine Estimator - background-normalized matched filter."""
    
    def __init__(self):
        super().__init__(name="ACE")
    
    def detect(self, image_data: np.ndarray, target_spectrum: np.ndarray) -> np.ndarray:
        height, width, n_bands = image_data.shape
        pixels = image_data.reshape(-1, n_bands)
        
        if not self._is_fitted:
            self.fit_background(pixels)
        
        target_centered = target_spectrum - self._background_mean
        pixels_centered = pixels - self._background_mean
        
        whitened_target = self._background_cov_inv @ target_centered
        numerator = (pixels_centered @ whitened_target) ** 2
        
        target_term = target_centered @ whitened_target
        whitened_pixels = pixels_centered @ self._background_cov_inv
        pixel_term = np.sum(whitened_pixels * pixels_centered, axis=1)
        
        denominator = np.maximum(target_term * pixel_term, 1e-10)
        ace_scores = np.clip(numerator / denominator, 0, 1)
        
        return ace_scores.reshape(height, width)
    
    @property
    def description(self) -> str:
        return "Adaptive Cosine Estimator - background-normalized detection"


# =============================================================================
# RXD Detector
# =============================================================================

class RXDDetector(BaseDetector):
    """Reed-Xiaoli Detector - anomaly detection using Mahalanobis distance."""
    
    def __init__(self):
        super().__init__(name="RXD")
    
    def detect(self, image_data: np.ndarray, target_spectrum: Optional[np.ndarray] = None) -> np.ndarray:
        height, width, n_bands = image_data.shape
        pixels = image_data.reshape(-1, n_bands)
        
        if not self._is_fitted:
            self.fit_background(pixels)
        
        pixels_centered = pixels - self._background_mean
        whitened = pixels_centered @ self._background_cov_inv
        rxd_scores = np.sum(whitened * pixels_centered, axis=1)
        
        # Weight by target similarity if provided
        if target_spectrum is not None:
            target_centered = target_spectrum - self._background_mean
            target_norm = np.linalg.norm(target_centered)
            pixel_norms = np.linalg.norm(pixels_centered, axis=1)
            
            cos_sim = np.dot(pixels_centered, target_centered) / (
                np.maximum(pixel_norms * target_norm, 1e-10))
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            similarity_weight = (cos_sim + 1) / 2
            rxd_scores = rxd_scores * similarity_weight
        
        return rxd_scores.reshape(height, width)
    
    @property
    def description(self) -> str:
        return "Reed-Xiaoli Detector - anomaly detection using Mahalanobis distance"


# =============================================================================
# CEM Detector
# =============================================================================

class CEMDetector(BaseDetector):
    """Constrained Energy Minimization detector."""
    
    def __init__(self):
        super().__init__(name="CEM")
    
    def detect(self, image_data: np.ndarray, target_spectrum: np.ndarray) -> np.ndarray:
        height, width, n_bands = image_data.shape
        pixels = image_data.reshape(-1, n_bands)
        
        if not self._is_fitted:
            self.fit_background(pixels)
        
        Rd = self._background_cov_inv @ target_spectrum
        dRd = max(target_spectrum @ Rd, 1e-10)
        filter_weights = Rd / dRd
        
        cem_scores = pixels @ filter_weights
        return cem_scores.reshape(height, width)
    
    @property
    def description(self) -> str:
        return "Constrained Energy Minimization - optimal linear filter"


# =============================================================================
# MF Detector (Matched Filter)
# =============================================================================

class MFDetector(BaseDetector):
    """Matched Filter detector - optimal linear filter for known target signature."""
    
    def __init__(self):
        super().__init__(name="MF")
    
    def detect(self, image_data: np.ndarray, target_spectrum: np.ndarray) -> np.ndarray:
        height, width, n_bands = image_data.shape
        pixels = image_data.reshape(-1, n_bands)
        
        if not self._is_fitted:
            self.fit_background(pixels)
        
        # Center the target spectrum
        target_centered = target_spectrum - self._background_mean
        
        # Compute the matched filter weights
        # MF = (C^-1 * t) / (t' * C^-1 * t)
        Ct = self._background_cov_inv @ target_centered
        tCt = max(target_centered @ Ct, 1e-10)
        mf_weights = Ct / tCt
        
        # Apply matched filter to all pixels (centered)
        pixels_centered = pixels - self._background_mean
        mf_scores = pixels_centered @ mf_weights
        
        return mf_scores.reshape(height, width)
    
    @property
    def description(self) -> str:
        return "Matched Filter - optimal linear filter for target detection"


# =============================================================================
# Detector Factory
# =============================================================================

DETECTORS = {
    'SAM': SAMDetector,
    'ACE': ACEDetector,
    'CEM': CEMDetector,
    'MF': MFDetector
}

# MLP-based models (dispatched to repo/Target_detection)
MLP_ALGORITHMS = {'MLP_AMF', 'MLP_ACE'}


def get_detector(name: str) -> BaseDetector:
    """Get detector instance by name."""
    name_upper = name.upper()
    if name_upper not in DETECTORS:
        raise ValueError(f"Unknown detector: {name}. Available: {list(DETECTORS.keys())}")
    return DETECTORS[name_upper]()


def get_available_detectors() -> List[Dict]:
    """Get list of available detectors with descriptions."""
    results = [
        {
            'id': name.lower(),
            'name': name,
            'description': DETECTORS[name]().description
        }
        for name in DETECTORS.keys()
    ]
    # Add MLP-based detectors
    results.append({'id': 'mlp_amf', 'name': 'MLP_AMF', 'description': 'Learnable MLP projector with AMF scoring'})
    results.append({'id': 'mlp_ace', 'name': 'MLP_ACE', 'description': 'Learnable MLP projector with ACE scoring'})
    return results


# =============================================================================
# Image Data Helper
# =============================================================================

class ImageDataLoader:
    """Helper class to load and manage image data for target detection."""
    
    def __init__(self, raster_path: str):
        self.raster_path = raster_path
        self.data: Optional[np.ndarray] = None
        self.transform = None
        self.crs = None
        self._load()
    
    def _load(self):
        """Load raster data."""
        with rasterio.open(self.raster_path) as src:
            # Read all bands and transpose to (H, W, Bands)
            data = src.read()
            self.data = np.transpose(data, (1, 2, 0)).astype(np.float32)
            self.transform = src.transform
            self.crs = src.crs
            self.bounds = src.bounds
            print(f"TARGET DETECTION - Raster CRS: {self.crs}")
            print(f"TARGET DETECTION - Raster bounds: {self.bounds}")
            print(f"TARGET DETECTION - Raster transform: {self.transform}")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape if self.data is not None else (0, 0, 0)
    
    @property
    def height(self) -> int:
        return self.shape[0]
    
    @property
    def width(self) -> int:
        return self.shape[1]
    
    @property
    def n_bands(self) -> int:
        return self.shape[2]
    
    @property
    def n_pixels(self) -> int:
        return self.height * self.width
    
    def get_spectrum_at_pixel(self, x: int, y: int) -> np.ndarray:
        """Get spectrum at pixel location (x, y)."""
        if self.data is None:
            raise ValueError("Data not loaded")
        y = max(0, min(y, self.height - 1))
        x = max(0, min(x, self.width - 1))
        return self.data[y, x, :].copy()
    
    def latlon_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon to pixel coordinates.
        
        Handles coordinate system transformation if raster is not in EPSG:4326.
        """
        from rasterio.transform import rowcol
        from pyproj import Transformer
        
        # Check if CRS is not WGS84 (EPSG:4326) and transform if needed
        if self.crs and str(self.crs).upper() != 'EPSG:4326':
            try:
                transformer = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
                print(f"TARGET DETECTION - Transformed ({lat}, {lon}) to ({x}, {y}) in {self.crs}")
            except Exception as e:
                print(f"TARGET DETECTION - Coordinate transform error: {e}")
                x, y = lon, lat
        else:
            x, y = lon, lat
        
        row, col = rowcol(self.transform, x, y)
        return int(col), int(row)
    
    def get_spectrum_at_latlon(self, lat: float, lon: float) -> np.ndarray:
        """Get spectrum at lat/lon coordinates."""
        x, y = self.latlon_to_pixel(lat, lon)
        return self.get_spectrum_at_pixel(x, y)
    
    def get_spectra_from_points(self, points: List[Tuple[int, int]]) -> np.ndarray:
        """Get spectra from multiple pixel points."""
        return np.array([self.get_spectrum_at_pixel(x, y) for x, y in points])
    
    def get_all_spectra(self) -> np.ndarray:
        """Get all spectra as 2D array (n_pixels, n_bands)."""
        return self.data.reshape(-1, self.n_bands)
    
    def get_random_background_spectra(self, n_samples: int,
                                       exclude_points: Optional[List[Tuple[int, int]]] = None,
                                       exclude_radius: int = 5) -> np.ndarray:
        """Get random background spectra excluding target regions."""
        valid_mask = np.ones((self.height, self.width), dtype=bool)
        
        if exclude_points:
            for x, y in exclude_points:
                y_min = max(0, y - exclude_radius)
                y_max = min(self.height, y + exclude_radius + 1)
                x_min = max(0, x - exclude_radius)
                x_max = min(self.width, x + exclude_radius + 1)
                valid_mask[y_min:y_max, x_min:x_max] = False
        
        valid_indices = np.argwhere(valid_mask)
        n_samples = min(n_samples, len(valid_indices))
        selected = valid_indices[np.random.choice(len(valid_indices), n_samples, replace=False)]
        
        return np.array([self.data[y, x, :] for y, x in selected])


# =============================================================================
# Optimal Threshold Finding
# =============================================================================

def find_optimal_threshold_otsu(data: np.ndarray) -> float:
    """Find optimal threshold using Otsu's method."""
    valid_data = data[np.isfinite(data)].flatten()
    if len(valid_data) == 0:
        return 0.5
    
    # Normalize to 0-255 for histogram
    min_val, max_val = valid_data.min(), valid_data.max()
    if max_val - min_val < 1e-10:
        return (min_val + max_val) / 2
    
    normalized = ((valid_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    # Compute histogram
    hist, bin_edges = np.histogram(normalized, bins=256, range=(0, 256))
    hist = hist.astype(float)
    
    # Total pixels
    total = hist.sum()
    if total == 0:
        return (min_val + max_val) / 2
    
    # Compute cumulative sums
    sum_all = np.dot(np.arange(256), hist)
    sum_b = 0
    w_b = 0
    
    max_variance = 0
    threshold_idx = 0
    
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        
        w_f = total - w_b
        if w_f == 0:
            break
        
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        
        # Between-class variance
        variance = w_b * w_f * (m_b - m_f) ** 2
        
        if variance > max_variance:
            max_variance = variance
            threshold_idx = i
    
    # Convert back to original scale
    optimal_threshold = min_val + (threshold_idx / 255.0) * (max_val - min_val)
    return float(optimal_threshold)


def find_optimal_threshold_target(data: np.ndarray, target_percentile: float = 90.0) -> float:
    """Find threshold that captures top percentile as targets."""
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        return 0.5
    return float(np.percentile(valid_data, target_percentile))


# =============================================================================
# Main Detection Function
# =============================================================================

def _run_mlp_detection(
    loader: 'ImageDataLoader',
    target_points_pixel: List[Tuple[int, int]],
    negative_points_pixel: List[Tuple[int, int]],
    algorithm: str,
    selected_bands: Optional[List[int]] = None,
    progress_callback=None,
) -> Dict:
    """Dispatch to repo/Target_detection MLP models.

    The MLP path trains a projector directly on CUDA. The shared
    `_model_cache` singleton + its `_cache_prepared` field aren't
    safe under concurrent calls, so we serialise the whole dispatch
    through the global GPU lock. This also prevents concurrent training
    from thrashing VRAM on a single low-spec GPU.
    """
    import sys
    import os
    repo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'repo', 'Target_detection')
    repo_path = os.path.abspath(repo_path)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from inference import detect as mlp_detect
    from .gpu_compute import _GPU_LOCK

    model_name = "new_method_mlp" if algorithm == "MLP_AMF" else "new_method_mlp_ace"
    cube = loader.data  # (H, W, C)

    print(f"TARGET DETECTION [MLP] - Running {model_name} on cube {cube.shape}")
    print(f"TARGET DETECTION [MLP] - pos_points={target_points_pixel}, neg_points={negative_points_pixel}")

    with _GPU_LOCK:
        result = mlp_detect(
            cube=cube,
            pos_points=target_points_pixel,
            neg_points=negative_points_pixel,
            model_name=model_name,
            threshold=None,  # auto (Otsu)
            progress_callback=progress_callback,
        )

    score_map = result["score_map"]
    mask = result["mask"]
    threshold = result.get("threshold", 0.5)
    loss_history = result.get("state", {}).get("projector_train_info", {}).get("loss_history", [])

    # Validate score_map shape matches the input cube's (H, W). A common
    # failure mode of the MLP path is returning a transposed or flattened
    # map, which silently produces an empty overlay when warped to AOI.
    expected_hw = cube.shape[:2]
    score_map = np.asarray(score_map)
    if score_map.shape != expected_hw:
        print(
            f"TARGET DETECTION [MLP] - WARNING score_map shape {score_map.shape} "
            f"!= expected {expected_hw}; attempting reshape/transpose"
        )
        if score_map.size == expected_hw[0] * expected_hw[1]:
            try:
                score_map = score_map.reshape(expected_hw)
            except Exception:
                pass
        elif score_map.shape[::-1] == expected_hw:
            score_map = score_map.T
    if score_map.shape != expected_hw:
        raise ValueError(
            f"MLP detection returned score_map shape {score_map.shape}; "
            f"expected {expected_hw}. Cannot warp to AOI."
        )

    finite_count = int(np.isfinite(score_map).sum())
    total = int(score_map.size)
    if finite_count == 0:
        raise ValueError(
            "MLP detection produced a score_map with no finite values "
            "(all NaN/Inf). Training likely failed."
        )
    if finite_count < total * 0.1:
        print(
            f"TARGET DETECTION [MLP] - WARNING only {finite_count}/{total} "
            f"pixels are finite"
        )
    print(
        f"TARGET DETECTION [MLP] - score_map OK shape={score_map.shape} "
        f"finite={finite_count}/{total} threshold={threshold}"
    )

    # Compute background stats for spectrum chart
    n_bg_samples = min(int(loader.n_pixels * 0.1), 10000)
    bg_spectra = loader.get_random_background_spectra(
        n_samples=n_bg_samples,
        exclude_points=target_points_pixel,
        exclude_radius=5,
    )
    target_spectra = loader.get_spectra_from_points(target_points_pixel)
    target_spectrum = np.mean(target_spectra, axis=0)

    valid_mask = np.isfinite(score_map)
    if np.any(valid_mask):
        min_val = float(np.min(score_map[valid_mask]))
        max_val = float(np.max(score_map[valid_mask]))
        mean_val = float(np.mean(score_map[valid_mask]))
    else:
        min_val, max_val, mean_val = 0.0, 1.0, 0.5

    used_bands = selected_bands if selected_bands else list(range(loader.n_bands))

    return {
        'detection_map': score_map,
        'threshold': float(threshold) if threshold is not None else find_optimal_threshold_otsu(score_map),
        'min_val': min_val,
        'max_val': max_val,
        'mean_val': mean_val,
        'transform': loader.transform,
        'crs': loader.crs,
        'target_spectrum': target_spectrum.tolist(),
        'background_mean': np.mean(bg_spectra, axis=0).tolist(),
        'background_std': np.std(bg_spectra, axis=0).tolist(),
        'algorithm': algorithm,
        'target_points_pixel': target_points_pixel,
        'used_bands': used_bands,
        'loss_history': loss_history,
    }


def run_target_detection(
    raster_path: str,
    target_points_latlon: List[Tuple[float, float]],
    negative_points_latlon: Optional[List[Tuple[float, float]]] = None,
    algorithm: str = 'SAM',
    threshold_percentile: float = 95.0,
    auto_threshold: bool = True,
    selected_bands: Optional[List[int]] = None,
    progress_callback=None,
) -> Dict:
    """
    Run target detection on a raster image.

    Args:
        raster_path: Path to the raster file
        target_points_latlon: List of (lat, lon) positive target coordinates
        negative_points_latlon: List of (lat, lon) negative (non-target) coordinates
        algorithm: Detection algorithm name
        threshold_percentile: Percentile for auto-threshold (if auto_threshold=False)
        auto_threshold: If True, use Otsu's method to find optimal threshold
        selected_bands: List of band indices to use (0-based). None means all bands.

    Returns:
        Dictionary with detection results
    """
    if negative_points_latlon is None:
        negative_points_latlon = []

    # Load image data
    loader = ImageDataLoader(raster_path)
    print(f"TARGET DETECTION - Loaded image: {loader.shape} (H x W x Bands)")

    # Hardcoded: exclude 60m bands (B1, B9) for Sentinel-2 cloud rasters.
    drop_s2_60m_bands(loader)
    print(f"TARGET DETECTION - After 60m drop: {loader.shape} (H x W x Bands)")

    # Apply band selection if specified
    if selected_bands is not None and len(selected_bands) > 0:
        print(f"TARGET DETECTION - Using selected bands: {selected_bands}")
        loader.data = loader.data[:, :, selected_bands]
        print(f"TARGET DETECTION - Filtered image shape: {loader.data.shape}")

    used_bands = selected_bands if selected_bands else list(range(loader.n_bands))

    # Convert lat/lon to pixel coordinates
    target_points_pixel = []
    for lat, lon in target_points_latlon:
        x, y = loader.latlon_to_pixel(lat, lon)
        target_points_pixel.append((x, y))
        print(f"TARGET DETECTION - Positive point ({lat}, {lon}) -> pixel ({x}, {y})")

    negative_points_pixel = []
    for lat, lon in negative_points_latlon:
        x, y = loader.latlon_to_pixel(lat, lon)
        negative_points_pixel.append((x, y))
        print(f"TARGET DETECTION - Negative point ({lat}, {lon}) -> pixel ({x}, {y})")

    # Dispatch to MLP models if requested
    algorithm_upper = algorithm.upper()
    if algorithm_upper in MLP_ALGORITHMS:
        return _run_mlp_detection(
            loader, target_points_pixel, negative_points_pixel,
            algorithm_upper, selected_bands, progress_callback,
        )

    # --- Classical detector path ---
    # Get target spectrum (average if multiple points)
    target_spectra = loader.get_spectra_from_points(target_points_pixel)
    target_spectrum = np.mean(target_spectra, axis=0)
    print(f"TARGET DETECTION - Target spectrum shape: {target_spectrum.shape}")
    print(f"TARGET DETECTION - Target spectrum values: {target_spectrum[:5]}... (first 5 bands)")

    # Get background spectra for fitting (exclude both positive and negative regions)
    all_exclude_points = target_points_pixel + negative_points_pixel
    n_bg_samples = min(int(loader.n_pixels * 0.1), 10000)
    bg_spectra = loader.get_random_background_spectra(
        n_samples=n_bg_samples,
        exclude_points=all_exclude_points,
        exclude_radius=5
    )
    print(f"TARGET DETECTION - Background samples: {bg_spectra.shape[0]}")

    # Get detector and run detection
    detector = get_detector(algorithm)
    detector.fit_background(bg_spectra)

    detection_map = detector.detect(loader.data, target_spectrum)

    # Normalize if needed (higher = target for all algorithms)
    if not detector._higher_is_target():
        max_val = np.max(detection_map)
        detection_map = max_val - detection_map

    # Calculate statistics
    valid_mask = np.isfinite(detection_map)
    if np.any(valid_mask):
        min_val = float(np.min(detection_map[valid_mask]))
        max_val = float(np.max(detection_map[valid_mask]))
        mean_val = float(np.mean(detection_map[valid_mask]))
    else:
        min_val, max_val, mean_val = 0.0, 1.0, 0.5

    # Find optimal threshold
    if auto_threshold:
        threshold = find_optimal_threshold_otsu(detection_map)
        print(f"TARGET DETECTION - Using Otsu's method, optimal threshold: {threshold:.4f}")
    else:
        threshold = float(np.percentile(detection_map[valid_mask], threshold_percentile))
        print(f"TARGET DETECTION - Using {threshold_percentile}th percentile threshold: {threshold:.4f}")

    print(f"TARGET DETECTION - Detection map range: [{min_val:.4f}, {max_val:.4f}]")
    print(f"TARGET DETECTION - Auto threshold ({threshold_percentile}th percentile): {threshold:.4f}")

    return {
        'detection_map': detection_map,
        'threshold': threshold,
        'min_val': min_val,
        'max_val': max_val,
        'mean_val': mean_val,
        'transform': loader.transform,
        'crs': loader.crs,
        'target_spectrum': target_spectrum.tolist(),
        'background_mean': np.mean(bg_spectra, axis=0).tolist(),
        'background_std': np.std(bg_spectra, axis=0).tolist(),
        'algorithm': algorithm,
        'target_points_pixel': target_points_pixel,
        'used_bands': used_bands
    }

