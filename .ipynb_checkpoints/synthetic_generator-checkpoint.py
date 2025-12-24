"""Synthetic TIFF stack generator for tethered strings of particles.

Produces a TIFF stack and a CSV ground-truth file describing particle locations
for each frame. Motion model:
- Frame 0: all particles start near their string center with small jitter (~3 px).
- Frames 1..20: strings stretch horizontally from the initial clustered position
  to their final relative offsets (linear interpolation).
- Frames 21..(20+100): particles remain at final stretched positions with
  small jitter (sigma=0.1 px) per frame.

Defaults follow the user's specification.
"""
from __future__ import annotations
import os
from typing import Sequence, Tuple
import numpy as np
import imageio.v3 as iio
import pandas as pd


def _gaussian_kernel(sigma: float, radius: int = None) -> np.ndarray:
    if radius is None:
        radius = int(np.ceil(4 * sigma))
    size = 2 * radius + 1
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    kern = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kern = kern / kern.max()
    return kern


def generate_strings_stack(
    out_tiff: str,
    out_csv: str,
    *,
    image_size: Tuple[int, int] = (512, 512),
    n_strings: int = 100,
    frames_pre: int = 1,
    frames_stretch: int = 20,
    frames_post: int = 100,
    offsets_a: Sequence[int] = (0, 6, 12, 18),
    offsets_b: Sequence[int] = (0, 6, 14, 20),
    psf_sigma: float = 4.0,
    intensity_range: Tuple[float, float] = (150.0, 500.0),
    background: float = 10.0,
    noise_std: float = 2.0,
    rng: np.random.Generator | None = None,
):
    rng = rng or np.random.default_rng(1)
    height, width = image_size
    total_frames = frames_pre + frames_stretch + frames_post

    # Create strings: half type A, half type B
    types = np.array(["A"] * (n_strings // 2) + ["B"] * (n_strings - n_strings // 2))
    rng.shuffle(types)

    # choose particle counts uniformly from 1..4
    counts = rng.choice([1, 2, 3, 4], size=n_strings)

    # choose string centers with margin to avoid clipping after stretching
    margin = int(max(offsets_b) + 10 + 4 * psf_sigma)
    xs = rng.integers(margin, width - margin, size=n_strings)
    ys = rng.integers(margin, height - margin, size=n_strings)
    centers = np.stack([xs, ys], axis=1).astype(float)

    # prepare particle records
    records = []
    particle_id = 0
    strings = []
    for sidx in range(n_strings):
        stype = types[sidx]
        offsets = np.array(offsets_a if stype == "A" else offsets_b)
        k = counts[sidx]
        # choose first k offsets (preserve ordering along string)
        chosen = offsets[:k]
        # each chosen offset becomes a particle with index along string
        strings.append({
            "string_id": sidx,
            "type": stype,
            "center": centers[sidx].copy(),
            "offsets": chosen,
        })

    # Precompute kernel
    kern = _gaussian_kernel(psf_sigma)
    krad = kern.shape[0] // 2

    # Prepare stack container (float32 for accumulation)
    stack = np.zeros((total_frames, height, width), dtype=np.float32)

    # For each particle build trajectories and add to stack
    for s in strings:
        cid = s["string_id"]
        cx, cy = s["center"]
        stype = s["type"]
        for pidx, off in enumerate(s["offsets"]):
            # final position relative to center (horizontal spacing)
            final_pos = np.array([cx + float(off), cy])
            # initial position clustered near center with +-3 px jitter
            init_pos = np.array([cx, cy]) + rng.normal(scale=3.0, size=2)
            # assign peak intensity
            intensity = float(rng.uniform(*intensity_range))

            # build frame-by-frame positions
            traj = np.zeros((total_frames, 2), dtype=float)
            # frame indices
            for t in range(total_frames):
                if t < frames_pre:
                    # initial cluster jitter (random walk around init_pos)
                    traj[t] = init_pos + rng.normal(scale=1.0, size=2)
                elif t < frames_pre + frames_stretch:
                    # linear interpolation to final_pos
                    tt = (t - frames_pre + 1) / frames_stretch
                    traj[t] = init_pos * (1 - tt) + final_pos * tt
                else:
                    # final position plus small diffusion +/-0.1px
                    traj[t] = final_pos + rng.normal(scale=0.1, size=2)

            # rasterize kernel for each frame
            for t in range(total_frames):
                x, y = traj[t]
                # integer center pixel
                cx_i = int(round(x))
                cy_i = int(round(y))
                x0 = cx_i - krad
                x1 = cx_i + krad + 1
                y0 = cy_i - krad
                y1 = cy_i + krad + 1
                kx0 = 0
                ky0 = 0
                kx1 = kern.shape[1]
                ky1 = kern.shape[0]
                # clip to image
                if x0 < 0:
                    kx0 = -x0
                    x0 = 0
                if y0 < 0:
                    ky0 = -y0
                    y0 = 0
                if x1 > width:
                    kx1 = kern.shape[1] - (x1 - width)
                    x1 = width
                if y1 > height:
                    ky1 = kern.shape[0] - (y1 - height)
                    y1 = height
                if x0 >= x1 or y0 >= y1:
                    continue
                stack[t, y0:y1, x0:x1] += intensity * kern[ky0:ky1, kx0:kx1]

            # record ground-truth per frame
            for t in range(total_frames):
                records.append(
                    {
                        "frame": int(t),
                        "x": float(traj[t, 0]),
                        "y": float(traj[t, 1]),
                        "intensity": float(intensity),
                        "string_id": int(cid),
                        "particle_index": int(pidx),
                        "offset_index": int(pidx),
                        "string_type": stype,
                    }
                )

    # add background and noise
    stack += background
    stack += rng.normal(scale=noise_std, size=stack.shape)

    # clip and convert to uint16 for saving
    stack = np.clip(stack, 0, 65535).astype(np.uint16)

    os.makedirs(os.path.dirname(out_tiff) or ".", exist_ok=True)
    # save TIFF stack
    iio.imwrite(out_tiff, stack, plugin="tifffile")

    # save CSV
    df = pd.DataFrame.from_records(records)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    return out_tiff, out_csv


if __name__ == "__main__":
    # quick CLI demo
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    tiff = os.path.join(out_dir, "synthetic_strings.tif")
    csvf = os.path.join(out_dir, "synthetic_strings_gt.csv")
    print("Generating synthetic dataset... this may take a few seconds")
    generate_strings_stack(tiff, csvf)
    print("Wrote:", tiff, csvf)
