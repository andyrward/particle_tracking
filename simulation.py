"""Run particle simulations and collect trajectories."""
from typing import List, Callable, Optional
import numpy as np
from particle import Particle
from integrator import euler_step


class Simulation:
    def __init__(self, particles: List[Particle], dt: float = 0.01, integrator: Callable = euler_step):
        self.particles = [p.copy() for p in particles]
        self.dt = dt
        self.integrator = integrator

    def step(self, force_fn: Callable[[Particle, float], np.ndarray]):
        self.integrator(self.particles, force_fn, self.dt)

    def run(self, n_steps: int, force_fn: Callable[[Particle, float], np.ndarray], record: bool = True):
        n = len(self.particles)
        dim = self.particles[0].pos.size if n > 0 else 2
        traj = np.zeros((n, n_steps + 1, dim)) if record else None
        if record:
            for i, p in enumerate(self.particles):
                traj[i, 0] = p.pos.copy()
        for t in range(1, n_steps + 1):
            self.step(force_fn)
            if record:
                for i, p in enumerate(self.particles):
                    traj[i, t] = p.pos.copy()
        return traj


def random_particles(n: int, box: float = 1.0, rng: Optional[np.random.Generator] = None):
    rng = rng or np.random.default_rng()
    particles = []
    for i in range(n):
        pos = rng.random(2) * box
        vel = np.zeros(2)
        particles.append(Particle(i, pos, vel, 1.0))
    return particles
