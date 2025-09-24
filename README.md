# record_video

PyQt5 based RealSense recording tool with optional NumPy analysis utilities.

## Repository layout

- `record/core/`: configuration dataclasses and shared helpers (`RecordConfig`,
  timestamped output directories, video helpers).
- `record/recording/`: RealSense capture threads built on top of `pyrealsense2`.
- `record/gui/`: PyQt main window, reusable threads and the embedded NPY viewer.
- `record/analysis/`: offline utilities such as `analyze_npy` producing CSV
  statistics from saved depth frames.
- `ui/`: Qt Designer `.ui` files for the GUI layout.
- `visual_npy.py`: lightweight CLI to preview `depth_raw_frames.npy` files via
  Matplotlib.

## Running the application

```bash
python -m record.gui.app
```

## Command line analysis

Generate per-frame statistics CSV files from recorded `depth_raw_frames.npy`
artifacts:

```bash
python -m record.analysis.npy path/to/depth_raw_frames.npy --out save/npy_analysis_out
```

## Cropping recorded sessions

Launch the GUI (`python -m record.gui.app`) and switch to the **Crop Frames** tab
to generate cropped datasets. You can either:

- Select previously saved outputs (`depth_raw_frames.npy` with optional color
  videos/images), choose the frame **start / end / step**, and export just that
  range of depth/color frames, or
- Load a RealSense `.bag` file directly and slice a specific frame range into
  per-frame depth/NPY artifacts.

Cropped artifacts are written to timestamped folders under
`save/crop_out/`, including aggregated `depth_raw_frames.npy`, per-frame
`.npy` files and any cropped image sequences. A `crop_info.json` file contains
the summary of each operation.

## Quick matplotlib preview

```bash
python visual_npy.py path/to/depth_raw_frames.npy --frame 0 --cmap viridis
```
