"""Detection helper utilities: detect_and_plot and minmass estimator.

Provides:
- `estimate_minmass(img, diameter, nsigma=3.0)` -> float
- `detect_and_plot(img, gt_df, diameter=5, separation=3, minmass=None, threshold=3.0, add_to_viewer=False, viewer=None, ax=None, show_plot=True)`

This module depends on `detect.detect_frame` (trackpy wrapper) and standard
scientific packages available in the project environment.
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def estimate_minmass(img: np.ndarray, diameter: int, nsigma: float = 3.0) -> float:
    """Estimate a sensible `minmass` threshold for trackpy.locate.

    The estimator uses a robust background (median) and MAD->sigma conversion,
    then scales by the PSF area and nsigma.
    """
    if img.size == 0:
        return 0.0
    bg = float(np.median(img))
    mad = float(np.median(np.abs(img - bg)))
    sigma = mad / 0.6745 if mad > 0 else float(np.std(img))
    area = np.pi * (diameter / 2.0) ** 2
    return bg * area + nsigma * sigma * np.sqrt(area)


def detect_and_plot(
    img: np.ndarray,
    gt_df: pd.DataFrame,
    diameter: int = 5,
    separation: int = 3,
    minmass: Optional[float] = None,
    threshold: float = 3.0,
    add_to_viewer: bool = False,
    viewer=None,
    ax=None,
    show_plot: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Run detection on `img`, evaluate against `gt_df`, and optionally plot.

    Returns (pts_det, metrics) where `pts_det` is a DataFrame containing
    detected coordinates and `metrics` is a dict with keys `tp`, `fp`, `fn`,
    and `mean_err` (localization error for matched detections).
    """
    from detect import detect_frame
    from scipy.spatial import cKDTree
    import matplotlib.pyplot as plt

    if minmass is None:
        minmass = estimate_minmass(img, diameter, nsigma=3.0)

    pts_det = detect_frame(img, method='trackpy', diameter=diameter, separation=separation, minmass=minmass)

    # evaluation
    def _evaluate_frame(gt_df_local: pd.DataFrame, det_df_local: pd.DataFrame, threshold_local: float = threshold):
        if len(det_df_local) == 0 or len(gt_df_local) == 0:
            return {'tp': 0, 'fp': len(det_df_local), 'fn': len(gt_df_local), 'mean_err': None}
        gt_pts = gt_df_local[['x', 'y']].values
        det_pts = det_df_local[['x', 'y']].values
        tree = cKDTree(gt_pts)
        dists, idx = tree.query(det_pts, k=1)
        matched = dists <= threshold_local
        tp = int(matched.sum())
        fp = int((~matched).sum())
        matched_gt = set(idx[matched].tolist())
        fn = int(len(gt_pts) - len(matched_gt))
        mean_err = float(dists[matched].mean()) if tp > 0 else None
        return {'tp': tp, 'fp': fp, 'fn': fn, 'mean_err': mean_err}

    metrics = _evaluate_frame(gt_df, pts_det, threshold_local=threshold)

    # plotting
    if show_plot:
        created_ax = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            created_ax = True
        ax.imshow(img, cmap='gray')
        ax.scatter(gt_df['x'], gt_df['y'], s=10, facecolors='none', edgecolors='r', label='GT')
        if len(pts_det):
            ax.scatter(pts_det['x'], pts_det['y'], s=10, facecolors='none', edgecolors='g', label='Det')
        ax.legend()
        if created_ax:
            plt.show()

    # optionally add to napari viewer
    if add_to_viewer and viewer is not None:
        try:
            viewer.add_points(pts_det[['x', 'y']].values, name='det', face_color='green')
        except Exception:
            pass

    return pts_det, metrics


def track_frames(
    stack: np.ndarray,
    start_frame: int,
    end_frame: int,
    diameter: int = 5,
    separation: int = 3,
    minmass: Optional[float] = None,
    verbose: bool = False,
    use_batch: bool = True,
    processes: Optional[int] = None,
) -> pd.DataFrame:
    """Detect particles on each frame in [start_frame, end_frame] and return a DataFrame of detections.

    Parameters
    - stack: 3D numpy array with shape (T, Y, X)
    - start_frame, end_frame: inclusive frame indices (0-based)
    - diameter, separation, minmass: forwarded to trackpy
    - verbose: print progress
    - use_batch: if True, use trackpy.batch (faster, multiprocessing); if False, use frame-by-frame loop
    - processes: number of processes for batch mode (default: os.cpu_count())

    Returns
    - pandas.DataFrame with columns ['frame','x','y','mass','signal'] (if available)
    """
    import os
    import trackpy as tp

    if stack.ndim != 3:
        raise ValueError("track_frames expects a 3D stack with shape (T, Y, X)")

    T = stack.shape[0]
    if start_frame < 0:
        start_frame = 0
    if end_frame >= T:
        end_frame = T - 1
    if end_frame < start_frame:
        raise ValueError("end_frame must be >= start_frame")

    # extract sub-stack
    sub_stack = stack[start_frame : end_frame + 1]

    if use_batch:
        if processes is None:
            processes = os.cpu_count() or 1
        if verbose:
            print(f"Running tp.batch with {processes} processes...")
        # Convert to float (same as detect_frame does for consistency)
        sub_stack_float = sub_stack.astype(float)
        # tp.batch returns a DataFrame with columns: x, y, mass, signal, frame
        out = tp.batch(sub_stack_float, diameter=diameter, minmass=minmass, separation=separation, processes=processes)
        # adjust frame numbers to match original stack indices
        out['frame'] = out['frame'] + start_frame
        if verbose:
            print(f"Batch detection complete: {len(out)} detections")
    else:
        rows = []
        from detect import detect_frame

        for f in range(start_frame, end_frame + 1):
            if verbose:
                print(f"Detecting frame {f}...")
            frame = stack[f]
            df = detect_frame(frame, method='trackpy', diameter=diameter, separation=separation, minmass=minmass)
            if 'frame' in df.columns:
                df['frame'] = f
            else:
                df = df.assign(frame=f)
            rows.append(df)

        if len(rows) == 0:
            return pd.DataFrame(columns=['frame', 'x', 'y', 'mass', 'signal'])

        out = pd.concat(rows, ignore_index=True, sort=False)

    # ensure column order
    cols = ['frame', 'x', 'y']
    for c in ['mass', 'signal']:
        if c in out.columns:
            cols.append(c)
    return out[cols].reset_index(drop=True)


def analyze_subpx_bias(
    tracks: pd.DataFrame,
    stack: np.ndarray,
    max_speed: Optional[float] = None,
) -> dict:
    """Analyze subpixel bias using trackpy's subpx_bias function.

    Parameters
    - tracks: DataFrame with columns ['frame','x','y'] (output of `track_frames`)
    - stack: 3D numpy array with shape (T, Y, X)
    - max_speed: optional maximum speed threshold for linking

    Returns
    - dict with keys: 'bias_x', 'bias_y', 'mean_bias' and diagnostic strings
    """
    import trackpy as tp

    if len(tracks) == 0:
        return {'bias_x': None, 'bias_y': None, 'mean_bias': None, 'note': 'No detections'}

    # tp.subpx_bias expects a DataFrame with 'x', 'y', 'frame' columns
    try:
        bias_result = tp.subpx_bias(tracks, max_speed=max_speed)
        # bias_result is typically a tuple (bias_x, bias_y)
        if isinstance(bias_result, tuple) and len(bias_result) == 2:
            bias_x, bias_y = bias_result
            mean_bias = np.sqrt(bias_x**2 + bias_y**2)
            return {
                'bias_x': float(bias_x),
                'bias_y': float(bias_y),
                'mean_bias': float(mean_bias),
                'note': 'Bias analysis complete'
            }
        else:
            # If the result is a different format, capture it
            return {
                'bias_x': None,
                'bias_y': None,
                'mean_bias': None,
                'note': f'Unexpected result format: {type(bias_result)}'
            }
    except Exception as e:
        return {
            'bias_x': None,
            'bias_y': None,
            'mean_bias': None,
            'note': f'Error during subpx_bias: {type(e).__name__}: {e}'
        }
