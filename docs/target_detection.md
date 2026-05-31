# Standalone Target Detection

`target_detection.py` is a single-file CLI for classical spectral target
detection. Copy that Python file to another directory, install the packages
below, and run it against a multiband GeoTIFF.

## Dependencies

```bash
pip install numpy rasterio
```

The script has no project-internal imports and does not require model weights.
It supports the classical `SAM`, `ACE`, `CEM`, and `MF` algorithms. The
trainable MLP target detectors are intentionally not bundled into this
single-file version.

## Input Data

The input must be a GeoTIFF with one or more spectral bands.

- Pixel prompts use `col,row` order.
- At least one positive point is required.
- Negative points are optional. They exclude nearby pixels from background
  sampling so obvious non-target prompts do not pollute the sampled context.
- `--bands` uses GeoTIFF 1-based band indexes. If it is omitted, every band in
  the GeoTIFF is used.

The script does not force a satellite-specific band policy. For reference, the
current platform Sentinel-2 flow uses the surface-oriented bands
`B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12` and omits `B1` and `B9` before target
detection.

## Run

```bash
python3 target_detection.py \
  --input /data/multiband_image.tif \
  --positive-point 120,88 \
  --positive-point 126,91 \
  --negative-point 40,52 \
  --algorithm SAM \
  --bands 1,2,3,4,5,6 \
  --output-dir /data/out
```

Useful options:

| Option | Meaning |
|---|---|
| `--algorithm` | `SAM`, `ACE`, `CEM`, or `MF`. Default is `SAM`. |
| `--threshold` | Manual score threshold. If omitted, Otsu thresholding is used. |
| `--background-samples` | Maximum random background spectra for covariance fitting. |
| `--exclude-radius` | Radius removed around positive and negative prompts. |

## Outputs

The output directory receives:

| File | Content |
|---|---|
| `target_detection_mask.tif` | Single-band binary mask with the input raster CRS and transform. |
| `target_detection_metadata.json` | Input path, bands, prompts, algorithm, threshold, and score statistics. |

## UI Input Placeholders

If this file is called from another UI layer, the button or map interaction can
be reduced to these variables before constructing the CLI command:

```python
positive_points_from_user = [(120, 88), (126, 91)]
negative_points_from_user = [(40, 52)]
selected_algorithm = "SAM"
selected_bands_1_based = [1, 2, 3, 4, 5, 6]
```
