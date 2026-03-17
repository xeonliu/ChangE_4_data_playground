#!/usr/bin/env python3
"""
pds4_to_heic.py — Convert Chang'E 4 PDS4 image files to HDR HEIC format.

Reads PDS4 label + binary data files from the CE4 mission (TCAM terrain
camera and PCAM panoramic camera), applies the appropriate processing for
each image type, and saves a 16-bit High Efficiency Image Container (HEIC)
suitable for HDR viewing.

Processing per image type
--------------------------
TCAM  (2C/2CL) : Array_3D_Image, already RGB — HDR linear stretch only.
PCAM colour (C): Array_2D_Image, RGGB Bayer — demosaic → HDR stretch.
PCAM pan   (Q) : Array_2D_Image, greyscale  — HDR stretch → replicate to RGB.

The output HEIC is encoded in "RGB;16" mode (16 bits per channel), which
pillow-heif stores inside a HEVC/HEIF container.  Most modern HEIC viewers
(Apple Photos, GNOME Image Viewer ≥ 43, Windows Photos) display the full
tonal range when hardware support is present.

Usage
-----
Single file:
    python pds4_to_heic.py input.2BL output.heic

Batch (glob):
    python pds4_to_heic.py --glob "PCAM/*.2BL" --output-dir heic_output/

Options:
    --stretch-low  FLOAT   Lower percentile for HDR stretch (default 0.5)
    --stretch-high FLOAT   Upper percentile for HDR stretch (default 99.5)
"""

import argparse
import glob as _glob
import sys
from pathlib import Path

import numpy as np
from skimage import exposure, img_as_float


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_pds4(label_path: str) -> np.ndarray:
    """
    Read a PDS4 label file and return the first data object as a float32
    array normalised to [0, 1].

    Parameters
    ----------
    label_path : str
        Path to the .2BL or .2CL PDS4 label file.

    Returns
    -------
    numpy.ndarray  (float32, shape H×W or H×W×3)
    """
    from pds4_tools import pds4_read  # import here so the module is optional
    data = pds4_read(str(label_path), quiet=True)
    raw = np.array(data[0].data)
    return img_as_float(raw).astype(np.float32)


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------

def debayer(img: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """
    Demosaic a 2-D Bayer-pattern image to an H×W×3 float32 RGB array.

    Uses the Menon (2007) algorithm — the same algorithm used in the
    original project notebooks.
    """
    from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
    rgb = demosaicing_CFA_Bayer_Menon2007(img, pattern)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def hdr_stretch(img: np.ndarray,
                low_pct: float = 0.5,
                high_pct: float = 99.5) -> np.ndarray:
    """
    Apply a gentle percentile linear stretch preserving the HDR tonal range.

    Compared with the 2 % stretch in the original notebooks, wider default
    percentiles (0.5 / 99.5) are used so that faint shadow and highlight
    detail is retained for HDR display.
    """
    p_lo = np.percentile(img, low_pct)
    p_hi = np.percentile(img, high_pct)
    if p_hi > p_lo:
        return exposure.rescale_intensity(
            img, in_range=(p_lo, p_hi)
        ).astype(np.float32)
    return img.copy()


def to_rgb_float(img: np.ndarray) -> np.ndarray:
    """
    Ensure *img* is a H×W×3 float32 array.

    * 3-D (H, W, 3) — returned as-is.
    * 2-D (H, W)    — replicated to three identical channels (greyscale→RGB).
    """
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    raise ValueError(f"Unexpected image shape: {img.shape}")


def float_to_uint16(img: np.ndarray) -> np.ndarray:
    """Convert a float [0, 1] image to uint16 [0, 65535]."""
    return (np.clip(img, 0.0, 1.0) * 65535).astype(np.uint16)


# ---------------------------------------------------------------------------
# HEIC output
# ---------------------------------------------------------------------------

def save_heic(img_float: np.ndarray, output_path: "Path | str", quality: int = 90) -> None:
    """
    Save a float [0, 1] H×W×3 image as a 16-bit HEIC (HDR) file.

    Parameters
    ----------
    img_float   : float32 H×W×3 array in [0, 1].
    output_path : destination path (str or Path).
    quality     : HEIF encoder quality hint (1–100, default 90).
    """
    import pillow_heif  # runtime import — keeps startup fast when not needed

    pillow_heif.register_heif_opener()

    img_u16 = float_to_uint16(to_rgb_float(img_float))
    h, w = img_u16.shape[:2]

    heif_file = pillow_heif.from_bytes(
        mode="RGB;16",
        size=(w, h),
        data=img_u16.tobytes(),
    )
    heif_file.save(str(output_path), quality=quality)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

# PCAM full-resolution Bayer dimensions (from actual CE4 data)
_PCAM_BAYER_SHAPES = {(1728, 2352)}


def _needs_debayer(img: np.ndarray, label_path: Path) -> bool:
    """
    Return True when the raw image is a 2-D Bayer-pattern array that needs
    demosaicing.

    Detection logic (in order of priority):
    1. If the image already has 3 channels it is already RGB — no demosaic.
    2. If the image shape matches known PCAM full-resolution Bayer dimensions,
       demosaic it.
    3. If the label path contains "PCAM" and the image mode is "C" (colour),
       demosaic it.
    4. Otherwise treat as panchromatic/greyscale.
    """
    if img.ndim == 3:
        return False  # already multi-band (e.g. TCAM)
    if img.shape in _PCAM_BAYER_SHAPES:
        return True
    # Heuristic: PCAM colour files have "-C-" in their filename
    name = label_path.stem.upper()
    if "PCAM" in name and "-C-" in name:
        return True
    return False


# ---------------------------------------------------------------------------
# Main conversion routine
# ---------------------------------------------------------------------------

def convert(label_path,
            output_path,
            stretch_low: float = 0.5,
            stretch_high: float = 99.5,
            quality: int = 90) -> Path:
    """
    Convert a single PDS4 label file to an HDR HEIC image.

    Parameters
    ----------
    label_path   : Path to the PDS4 label (.2BL or .2CL).
    output_path  : Destination HEIC file path.
    stretch_low  : Lower percentile for HDR stretch.
    stretch_high : Upper percentile for HDR stretch.
    quality      : HEIF encoder quality (1–100).

    Returns
    -------
    Path  — the written output file.
    """
    label_path = Path(label_path)
    output_path = Path(output_path)

    print(f"[read]    {label_path}")
    img = read_pds4(label_path)
    print(f"          shape={img.shape}  dtype={img.dtype}  "
          f"range=[{img.min():.4f}, {img.max():.4f}]")

    if _needs_debayer(img, label_path):
        print("[debayer] Applying Menon 2007 RGGB demosaicing …")
        img = debayer(img)
        print(f"          → shape={img.shape}")

    # Ensure float RGB before stretch
    img = to_rgb_float(img)

    print(f"[stretch] {stretch_low:.1f}–{stretch_high:.1f} percentile HDR stretch …")
    img = hdr_stretch(img, stretch_low, stretch_high)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[save]    {output_path}  (16-bit HEIC, quality={quality})")
    save_heic(img, output_path, quality=quality)

    size_kb = output_path.stat().st_size / 1024
    print(f"          ✓  {size_kb:.1f} KB written\n")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert Chang'E 4 PDS4 image files (TCAM/PCAM) to 16-bit HDR HEIC."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", nargs="?",
                   help="Single PDS4 label file (.2BL or .2CL)")
    p.add_argument("output", nargs="?",
                   help="Output HEIC file path")
    p.add_argument("--glob", metavar="PATTERN",
                   help='Glob pattern for batch conversion, e.g. "PCAM/*.2BL"')
    p.add_argument("--output-dir", metavar="DIR", default="heic_output",
                   help="Output directory for batch conversion")
    p.add_argument("--stretch-low", type=float, default=0.5,
                   help="Lower percentile for HDR stretch")
    p.add_argument("--stretch-high", type=float, default=99.5,
                   help="Upper percentile for HDR stretch")
    p.add_argument("--quality", type=int, default=90,
                   help="HEIF encoder quality (1–100)")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    kw = dict(
        stretch_low=args.stretch_low,
        stretch_high=args.stretch_high,
        quality=args.quality,
    )

    if args.glob:
        files = sorted(_glob.glob(args.glob))
        if not files:
            print(f"ERROR: no files found matching '{args.glob}'", file=sys.stderr)
            return 1
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        errors = 0
        for f in files:
            src = Path(f)
            dst = out_dir / (src.stem + ".heic")
            try:
                convert(src, dst, **kw)
            except Exception as exc:
                print(f"ERROR converting {f}: {exc}", file=sys.stderr)
                errors += 1
        return errors

    if args.input and args.output:
        try:
            convert(args.input, args.output, **kw)
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    _build_parser().print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
