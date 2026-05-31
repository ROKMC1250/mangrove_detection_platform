# Standalone SAM3 Segmentation

`sam3.py` is a standalone CLI for SAM3 image segmentation from GeoTIFF input.
It contains GeoTIFF RGB extraction, SAM3 image encoding, point prompting, text
prompting, and mask writing in the same file. The default checkpoint bundle is
kept beside it under `checkpoints/sam3/`.

## Dependencies

Install raster/image dependencies and a PyTorch build that matches the target
machine first, then install SAM3:

```bash
pip install numpy rasterio pillow torch torchvision
pip install "git+https://github.com/facebookresearch/sam3.git"
```

The bundled assets expected by the default command are:

| Relative file | Purpose |
|---|---|
| `checkpoints/sam3/sam3.pt` | SAM3 model weights. |
| `checkpoints/sam3/config.json` | Hugging Face model config shipped with the checkpoint set. |
| `checkpoints/sam3/bpe_simple_vocab_16e6.txt.gz` | SAM3 tokenizer vocabulary. |

The script passes the bundled `sam3.pt` and BPE file directly to the SAM3
builder, so it does not rely on a Hugging Face cache for the default run. To
swap checkpoint sets, pass another directory containing `sam3.pt` with
`--checkpoint-dir`; use `--bpe-path` only when that directory does not carry
the BPE asset.

## Input Data

SAM3 receives an RGB image, not the entire spectral cube. The script accepts a
GeoTIFF and converts three selected bands into an 8-bit RGB image.

- `--rgb-bands` uses GeoTIFF 1-based band indexes in `R,G,B` order.
- The CLI default is `3,2,1`, matching a common Sentinel-2 raster ordered as
  blue, green, red for the first three visible bands.
- Point prompts use `col,row` pixel coordinates.
- Use either point mode or text mode for one run.

## Point Mode

```bash
python3 sam3.py \
  --input /data/rgb_or_multiband.tif \
  --rgb-bands 3,2,1 \
  --positive-point 320,220 \
  --negative-point 280,210 \
  --output-dir /data/out
```

At least one `--positive-point` is required in point mode. Negative points are
optional.

## Text Mode

```bash
python3 sam3.py \
  --input /data/rgb_or_multiband.tif \
  --rgb-bands 3,2,1 \
  --text-prompt water \
  --score-threshold 0.5 \
  --output-dir /data/out
```

Text mode writes one mask for each retained SAM3 instance whose score passes
`--score-threshold`.

## Outputs

Point mode writes:

| File | Content |
|---|---|
| `sam3_mask.tif` | Best single binary mask for the point prompts. |
| `sam3_metadata.json` | RGB bands, prompts, score, device, and mask path. |

Text mode writes:

| File | Content |
|---|---|
| `sam3_instance_001.tif`, ... | Binary instance masks retained from the text prompt. |
| `sam3_metadata.json` | Prompt, score threshold, boxes, scores, and mask paths. |

Every mask preserves the input GeoTIFF CRS, transform, width, and height.

Keep the `checkpoints/sam3/` directory beside `sam3.py` when moving this
bundle. If it is missing, provide an equivalent checkpoint set with
`--checkpoint-dir`.

## UI Input Placeholders

```python
positive_points_from_user = [(320, 220)]
negative_points_from_user = [(280, 210)]
text_prompt_from_user = "water"
score_threshold_from_user = 0.5
rgb_bands_1_based = [3, 2, 1]
```
