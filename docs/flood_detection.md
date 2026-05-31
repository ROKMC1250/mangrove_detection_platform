# Standalone Flood Detection

`flood_detection.py` is a standalone CLI for Sentinel-1 VV flood segmentation.
It creates the UNet++ model, loads the checkpoint, normalizes the SAR raster,
runs tiled inference, and writes the binary flood mask itself. The default
checkpoint bundle is kept beside it under `checkpoints/flood_unetpp_vh_v1/`.

## Dependencies

```bash
pip install numpy rasterio torch segmentation-models-pytorch
```

The bundled assets expected by the default command are:

| Relative file | Purpose |
|---|---|
| `checkpoints/flood_unetpp_vh_v1/best.pt` | Flood UNet++ weights. |
| `checkpoints/flood_unetpp_vh_v1/config.yaml` | Training/model config kept with the checkpoint. |

The checkpoint is expected to contain weights for `UnetPlusPlus` with a
`resnet34` encoder, 3 input channels, and 1 output class. To replace it, pass
another compatible weight file with `--checkpoint`.

## Input Data

The input must be a GeoTIFF with Sentinel-1 VV dB values in band 1.

- Band 1 is clipped from `-30 dB` to `10 dB` and normalized to `[0, 1]`.
- The normalized VV array is repeated into 3 model channels.
- A GeoTIFF may also contain VH in band 2, but this standalone inference path
  does not use VH.
- No UI geometry or map cache is required. Give the SAR file path directly.

## Run

```bash
python3 flood_detection.py \
  --input /data/sentinel1_vv_vh.tif \
  --threshold 0.5 \
  --output-dir /data/out
```

Useful options:

| Option | Meaning |
|---|---|
| `--checkpoint` | Optional replacement checkpoint path. Defaults to the bundled docs weight. |
| `--threshold` | Probability threshold for the binary flood mask. Default is `0.5`. |
| `--patch-size` | Tiled inference patch size. Default is `512`. |
| `--overlap` | Overlapping pixels between adjacent patches. Default is `128`. |
| `--batch-size` | Patch batch size. Default is `4`. |

## Outputs

The output directory receives:

| File | Content |
|---|---|
| `flood_mask.tif` | Single-band binary flood mask with the input raster CRS and transform. |
| `flood_metadata.json` | Checkpoint path, threshold, tile parameters, band policy, and probability statistics. |

## UI Input Placeholders

```python
input_sar_path = "/data/sentinel1_vv_vh.tif"
checkpoint_path = "checkpoints/flood_unetpp_vh_v1/best.pt"
threshold_from_user = 0.5
```

Keep the `checkpoints/flood_unetpp_vh_v1/` directory beside
`flood_detection.py` when moving this bundle.
