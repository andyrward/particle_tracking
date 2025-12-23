"""Simple integrators and force utilities for particle simulation."""
from typing import Callable, List
import numpy as np
from particle import Particle


def brownian_force(sigma: float, dim: int = 2, rng: np.random.Generator = None):
    rng = rng or np.random.default_rng()
    return rng.normal(scale=sigma, size=(dim,))


def euler_step(particles: List[Particle], force_fn: Callable[[Particle, float], np.ndarray], dt: float):
    for p in particles:
        f = force_fn(p, dt)
        a = f / p.mass
        p.vel = p.vel + a * dt
        p.pos = p.pos + p.vel * dt


def verlet_step(particles: List[Particle], force_fn: Callable[[Particle, float], np.ndarray], dt: float, prev_accs=None):
    # Very small helper Verlet implementation for identical-mass particles.
    if prev_accs is None:
        prev_accs = [np.zeros_like(p.pos) for p in particles]
    new_accs = []
    for i, p in enumerate(particles):
        f = force_fn(p, dt)
        a = f / p.mass
        p.pos = p.pos + p.vel * dt + 0.5 * a * dt * dt
        new_accs.append(a)
    for i, p in enumerate(particles):
        p.vel = p.vel + 0.5 * (prev_accs[i] + new_accs[i]) * dt
    return new_accs
