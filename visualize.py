"""Visualization helpers for particle trajectories."""
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_trajectories(traj: np.ndarray, ax: Optional[plt.Axes] = None, figsize=(6, 6), title: Optional[str] = None):
    # traj shape: (n_particles, n_steps, dim)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    n, steps, dim = traj.shape
    for i in range(n):
        ax.plot(traj[i, :, 0], traj[i, :, 1], lw=1)
        ax.scatter(traj[i, 0, 0], traj[i, 0, 1], marker="o", s=20)
        ax.scatter(traj[i, -1, 0], traj[i, -1, 1], marker="x", s=20)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    return ax
