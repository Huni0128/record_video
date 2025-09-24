"""Quick matplotlib preview for ``depth_raw_frames.npy`` files."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npy_path", type=Path, help="Path to a depth .npy file")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to visualize when the array is 3-D (default: 0)",
    )
    parser.add_argument(
        "--cmap",
        default="gray",
        help="Matplotlib colour map (default: gray)",
    )
    return parser.parse_args()


def load_frame(array: np.ndarray, idx: int) -> np.ndarray:
    """Return a 2-D frame regardless of the array dimensionality."""
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        idx = max(0, min(idx, array.shape[0] - 1))
        return array[idx]
    raise ValueError(f"Unsupported array shape: {array.shape}")


def main() -> None:
    args = parse_args()
    array = np.load(args.npy_path)
    frame = load_frame(array, args.frame)

    plt.imshow(frame, cmap=args.cmap)
    plt.title(f"{args.npy_path.name} (frame {args.frame})")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
