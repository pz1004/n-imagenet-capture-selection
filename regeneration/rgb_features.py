"""Recompute the whole-photograph orientation covariate from authorized ImageNet JPEGs."""
import argparse
import csv
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path(__file__).resolve().parents[1] / "data"


def rgb_to_log_luminance(rgb):
    """Linearize canonical byte sRGB and log Rec. 709 linear-light luminance."""
    pixels = np.asarray(rgb)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("Expected an H x W x 3 uint8 sRGB image")
    return np.log(np.maximum(
        RGB_LUT[0, pixels[:, :, 0]] + RGB_LUT[1, pixels[:, :, 1]] + RGB_LUT[2, pixels[:, :, 2]],
        1e-4,
    ))

def make_lut():
    encoded = np.arange(256, dtype=np.float64)/255.0
    linear = np.where(encoded <= 0.04045, encoded/12.92, ((encoded+0.055)/1.055)**2.4)
    return np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)[:, None]*linear

RGB_LUT = make_lut()

def energy_on_log_grid(log_y, valid_mask=None):
    """Compute the two mean squared central differences on the same valid pixels.

    For an optional region-of-interest mask, requiring both
    horizontal and both vertical neighbours additionally excludes derivative stencils
    which would cross the ROI boundary, including holes.
    """
    y = np.asarray(log_y, dtype=np.float64)
    if y.ndim != 2 or min(y.shape) < 3:
        raise ValueError("No common 3 x 3 derivative interior")
    dx = (y[1:-1, 2:] - y[1:-1, :-2]) * 0.5
    dy = (y[2:, 1:-1] - y[:-2, 1:-1]) * 0.5
    if valid_mask is not None:
        ok = np.asarray(valid_mask, dtype=bool)
        if ok.shape != y.shape:
            raise ValueError("Valid footprint shape mismatch")
        centers = ok[1:-1, 1:-1] & ok[1:-1, 2:] & ok[1:-1, :-2] & ok[2:, 1:-1] & ok[:-2, 1:-1]
        dx, dy = dx[centers], dy[centers]
    if not dx.size:
        raise ValueError("No valid derivative stencil in common footprint")
    ex = float(np.mean(dx*dx, dtype=np.float64))
    ey = float(np.mean(dy*dy, dtype=np.float64))
    orientation = (ey-ex)/(ey+ex+1e-12)
    if not np.isfinite(orientation):
        raise ValueError("Nonfinite orientation")
    return {"energy_x": ex, "energy_y": ey, "orientation": orientation, "n_derivative_pixels": int(dx.size)}

def energy_on_grid(rgb, valid_mask=None):
    return energy_on_log_grid(rgb_to_log_luminance(rgb), valid_mask)

def whole_source(im):
    w, h = im.size
    long_side = max(w, h)
    shape = tuple(max(1, round(x*640/long_side)) for x in (w, h))
    # Pillow does not auto-apply JPEG EXIF orientation. Do not invoke exif_transpose.
    rendered = im.convert("RGB").resize(shape, resample=Image.Resampling.LANCZOS)
    result = energy_on_grid(rendered)
    return dict(result, width_px=w, height_px=h, feature_width_px=shape[0], feature_height_px=shape[1])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory containing the original validation JPEG filenames")
    parser.add_argument("--output", type=Path, required=True, help="New CSV file")
    parser.add_argument("--source-id", help="Limit to one synset/filename-stem")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The output must be a new file")
    with (DATA / "source_index.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["cohort"] == "full" and
                (args.source_id is None or r["source_id"] == args.source_id)]
    if not rows:
        parser.error("Unknown confirmation source")
    rows.sort(key=lambda r: (int(r["label"]), r["source_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as scratch:
        temp = Path(scratch)/"features.csv"
        with temp.open("w", newline="") as f:
            fields = ["source_id", "synset", "label", "split", "energy_x", "energy_y", "orientation",
                      "n_derivative_pixels", "width_px", "height_px", "feature_width_px", "feature_height_px"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                image_path = args.images / (row["source_id"].split("/")[1] + ".JPEG")
                with Image.open(image_path) as image:
                    values = whole_source(image)
                writer.writerow({k: row[k] for k in fields[:4]} | values)
        temp.rename(args.output)


if __name__ == "__main__":
    main()
