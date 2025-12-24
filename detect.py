"""Lightweight detection wrapper using trackpy.

Provides `detect_frame(frame, method='trackpy', **params)` which returns
a pandas DataFrame with columns ['x','y','mass','signal','frame'].

If `trackpy` is not available, raises ImportError.
"""
from typing import Optional

import numpy as np
import pandas as pd


def detect_frame(frame: np.ndarray, method: str = "trackpy", *, diameter: int = 5, separation: int = 3, minmass: Optional[float] = None, **kwargs) -> pd.DataFrame:
    """Detect particle centers in a single 2D image.

    Parameters
    - frame: 2D numpy array (grayscale image)
    - method: currently only 'trackpy' supported
    - diameter: approximate particle diameter in pixels (passed to trackpy.locate)
    - separation: minimum separation between particles
    - minmass: minimum integrated brightness (optional)
    - kwargs: forwarded to the detector

    Returns
    - pandas.DataFrame with at least columns ['x','y','mass','signal','frame']
    """
    if frame.ndim != 2:
        raise ValueError("detect_frame expects a 2D image (single frame)")

    if method != "trackpy":
        raise NotImplementedError(f"Only 'trackpy' method is implemented, got: {method}")

    try:
        import trackpy as tp
    except Exception as e:
        raise ImportError("trackpy is required for detect_frame(method='trackpy'). Install it in your environment.") from e

    # trackpy expects a floating image for some routines
    img = frame.astype(float)

    # call trackpy.locate; keep results as a DataFrame
    f = tp.locate(img, diameter=diameter, minmass=minmass, separation=separation, **kwargs)

    # Ensure required columns exist
    df = f.copy()
    # trackpy usually returns 'x','y','mass' and may include 'peak_value' or 'signal'
    if "signal" not in df.columns:
        if "peak_value" in df.columns:
            df["signal"] = df["peak_value"]
        else:
            df["signal"] = df.get("mass", np.nan)

    # add 'frame' column default 0 (caller can set differently when linking)
    df["frame"] = 0

    # keep a minimal set of columns (but preserve extras)
    cols = [c for c in ["x", "y", "mass", "signal", "frame"] if c in df.columns]

    return df[cols].reset_index(drop=True)
