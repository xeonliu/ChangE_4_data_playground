#!/usr/bin/env python3
"""
generate_sample_pds4.py — Create synthetic Chang'E 4 PDS4 test files.

Generates small but format-accurate PDS4 files that mimic the three
image types produced by the CE4 mission:

  - TCAM   : Array_3D_Image (H × W × 3), uint8, terrain camera colour
  - PCAM-C : Array_2D_Image (H × W),     uint8, close-up camera Bayer RGGB
  - PCAM-Q : Array_2D_Image (H × W),     uint8, panoramic panchromatic

Run:
    python generate_sample_pds4.py [--output-dir SAMPLE_DIR]
"""

import argparse
import textwrap
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# PDS4 XML helpers
# ---------------------------------------------------------------------------

_PDS4_HEADER = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
""")

_PDS4_IDENT = textwrap.dedent("""\
      <Identification_Area>
        <logical_identifier>{lid}</logical_identifier>
        <version_id>1.0</version_id>
        <title>Chang'E 4(CE-4) mission</title>
        <information_model_version>1.5.0.0</information_model_version>
        <product_class>Product_Observational</product_class>
        <Modification_History>
          <Modification_Detail>
            <modification_date>2019-01-03</modification_date>
            <version_id>1.0</version_id>
            <description>None</description>
          </Modification_Detail>
        </Modification_History>
      </Identification_Area>
""")

_PDS4_OBS = textwrap.dedent("""\
      <Observation_Area>
        <Time_Coordinates>
          <start_date_time>{start}Z</start_date_time>
          <stop_date_time>{stop}Z</stop_date_time>
        </Time_Coordinates>
        <Primary_Result_Summary>
          <purpose>Science</purpose>
          <processing_level>Calibrated</processing_level>
        </Primary_Result_Summary>
        <Investigation_Area>
          <name>CE4</name>
          <type>Mission</type>
          <Internal_Reference>
            <lid_reference>urn:test:changee4:mission</lid_reference>
            <reference_type>data_to_investigation</reference_type>
          </Internal_Reference>
        </Investigation_Area>
        <Observing_System>
          <Observing_System_Component>
            <name>CE4La</name>
            <type>Spacecraft</type>
          </Observing_System_Component>
        </Observing_System>
        <Target_Identification>
          <name>Lunar</name>
          <type>Satellite</type>
        </Target_Identification>
      </Observation_Area>
""")


def _array_3d_label(data_file: str, height: int, width: int, bands: int) -> str:
    """PDS4 label for an Array_3D_Image (TCAM)."""
    return (
        _PDS4_HEADER
        + _PDS4_IDENT.format(lid=f"urn:test:changee4:{Path(data_file).stem}")
        + _PDS4_OBS.format(
            start="2019-01-11T19:57:09.000",
            stop="2019-01-11T19:57:09.000",
        )
        + textwrap.dedent(f"""\
      <File_Area_Observational>
        <File>
          <file_name>{data_file}</file_name>
        </File>
        <Array_3D_Image>
          <offset unit="byte">0</offset>
          <axes>3</axes>
          <axis_index_order>Last Index Fastest</axis_index_order>
          <Element_Array>
            <data_type>UnsignedByte</data_type>
            <unit>data number</unit>
          </Element_Array>
          <Axis_Array>
            <axis_name>Line</axis_name>
            <elements>{height}</elements>
            <sequence_number>1</sequence_number>
          </Axis_Array>
          <Axis_Array>
            <axis_name>Sample</axis_name>
            <elements>{width}</elements>
            <sequence_number>2</sequence_number>
          </Axis_Array>
          <Axis_Array>
            <axis_name>Band</axis_name>
            <elements>{bands}</elements>
            <sequence_number>3</sequence_number>
          </Axis_Array>
        </Array_3D_Image>
      </File_Area_Observational>
    </Product_Observational>
    """)
    )


def _array_2d_label(data_file: str, height: int, width: int) -> str:
    """PDS4 label for an Array_2D_Image (PCAM Bayer or panchromatic)."""
    return (
        _PDS4_HEADER
        + _PDS4_IDENT.format(lid=f"urn:test:changee4:{Path(data_file).stem}")
        + _PDS4_OBS.format(
            start="2019-01-04T08:45:59.000",
            stop="2019-01-04T08:45:59.000",
        )
        + textwrap.dedent(f"""\
      <File_Area_Observational>
        <File>
          <file_name>{data_file}</file_name>
        </File>
        <Array_2D_Image>
          <offset unit="byte">0</offset>
          <axes>2</axes>
          <axis_index_order>Last Index Fastest</axis_index_order>
          <Element_Array>
            <data_type>UnsignedByte</data_type>
            <unit>data number</unit>
          </Element_Array>
          <Axis_Array>
            <axis_name>Line</axis_name>
            <elements>{height}</elements>
            <sequence_number>1</sequence_number>
          </Axis_Array>
          <Axis_Array>
            <axis_name>Sample</axis_name>
            <elements>{width}</elements>
            <sequence_number>2</sequence_number>
          </Axis_Array>
        </Array_2D_Image>
      </File_Area_Observational>
    </Product_Observational>
    """)
    )


# ---------------------------------------------------------------------------
# Image data generators (realistic synthetic content)
# ---------------------------------------------------------------------------

def _make_tcam(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate a TCAM terrain image: greyscale lunar surface with mild colour
    tint and noise.  Shape = (height, width, 3), uint8.
    """
    # Base luminance: gradient across image simulating uneven illumination
    base = np.linspace(80, 200, width, dtype=np.float32)
    base = np.tile(base, (height, 1))
    # Add some boulders/craters as darker spots
    for _ in range(5):
        cy, cx = rng.integers(0, height), rng.integers(0, width)
        r = rng.integers(3, max(4, min(height, width) // 6))
        yy, xx = np.ogrid[:height, :width]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        base[mask] *= rng.uniform(0.5, 0.85)
    base += rng.normal(0, 4, base.shape)
    base = np.clip(base, 0, 255)
    # Slight warm tint (more red, less blue) typical of sunlit lunar surface
    rgb = np.stack([base * 1.05, base * 1.00, base * 0.88], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _make_pcam_bayer(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate a PCAM colour raw image: RGGB Bayer-pattern 2D array, uint8.
    Shape = (height, width).
    """
    # Build separate channel planes then interleave into RGGB mosaic
    r_plane = rng.integers(100, 180, (height // 2, width // 2), dtype=np.uint8)
    g_plane = rng.integers(130, 210, (height // 2, width // 2), dtype=np.uint8)
    b_plane = rng.integers(80,  160, (height // 2, width // 2), dtype=np.uint8)

    bayer = np.zeros((height, width), dtype=np.uint8)
    bayer[0::2, 0::2] = r_plane               # R
    bayer[0::2, 1::2] = g_plane               # Gr
    bayer[1::2, 0::2] = g_plane               # Gb
    bayer[1::2, 1::2] = b_plane               # B
    return bayer


def _make_pcam_pan(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate a PCAM panchromatic image: greyscale, uint8.
    Shape = (height, width).
    """
    base = np.linspace(60, 220, width, dtype=np.float32)
    base = np.tile(base, (height, 1))
    base += rng.normal(0, 6, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_samples(output_dir: Path, height: int = 60, width: int = 80,
                     seed: int = 42) -> dict:
    """
    Write three synthetic CE4 PDS4 files to *output_dir*.

    Returns a dict mapping descriptive keys to PDS4 label paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    files_written = {}

    # 1) TCAM — terrain camera (already-demosaiced RGB)
    tcam_data_name = "CE4_GRAS_TCAM-I-065_SCI_N_20190111190315_20190111190315_0009_A.2C"
    tcam_label_name = tcam_data_name.replace(".2C", ".2CL")
    arr = _make_tcam(height, width, rng)
    (output_dir / tcam_data_name).write_bytes(arr.tobytes())
    (output_dir / tcam_label_name).write_text(
        _array_3d_label(tcam_data_name, height, width, 3)
    )
    files_written["tcam"] = output_dir / tcam_label_name
    print(f"  TCAM  : {output_dir / tcam_label_name}  shape={arr.shape}")

    # 2) PCAM colour — close-up camera (Bayer RGGB, needs demosaicing)
    # Dimensions must be even for Bayer pattern
    bh, bw = (height // 2) * 2, (width // 2) * 2
    pcamc_data_name = "CE4_GRAS_PCAML-C-006_SCI_N_20190104084559_20190104084559_0001_B.2B"
    pcamc_label_name = pcamc_data_name.replace(".2B", ".2BL")
    arr = _make_pcam_bayer(bh, bw, rng)
    (output_dir / pcamc_data_name).write_bytes(arr.tobytes())
    (output_dir / pcamc_label_name).write_text(
        _array_2d_label(pcamc_data_name, bh, bw)
    )
    files_written["pcam_color"] = output_dir / pcamc_label_name
    print(f"  PCAM-C: {output_dir / pcamc_label_name}  shape={arr.shape}")

    # 3) PCAM panchromatic — Q-mode (greyscale, no demosaicing)
    pcamq_data_name = "CE4_GRAS_PCAMR-Q-032_SCI_N_20190112082618_20190112082618_0003_B.2B"
    pcamq_label_name = pcamq_data_name.replace(".2B", ".2BL")
    arr = _make_pcam_pan(height, width, rng)
    (output_dir / pcamq_data_name).write_bytes(arr.tobytes())
    (output_dir / pcamq_label_name).write_text(
        _array_2d_label(pcamq_data_name, height, width)
    )
    files_written["pcam_pan"] = output_dir / pcamq_label_name
    print(f"  PCAM-Q: {output_dir / pcamq_label_name}  shape={arr.shape}")

    return files_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CE4 PDS4 sample files for testing."
    )
    parser.add_argument(
        "--output-dir", default="sample_data",
        help="Directory to write sample files (default: sample_data/)"
    )
    parser.add_argument(
        "--height", type=int, default=60,
        help="Image height in pixels (default: 60)"
    )
    parser.add_argument(
        "--width", type=int, default=80,
        help="Image width in pixels (default: 80)"
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    print(f"Generating sample PDS4 files in: {out}")
    generate_samples(out, height=args.height, width=args.width)
    print("Done.")


if __name__ == "__main__":
    main()
