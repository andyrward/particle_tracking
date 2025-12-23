from dataclasses import dataclass, field
import numpy as np
from typing import Sequence


@dataclass
class Particle:
    """Simple 2D particle state container."""
    id: int
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    mass: float = 1.0

    def copy(self):
        return Particle(self.id, self.pos.copy(), self.vel.copy(), self.mass)

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([self.pos, self.vel])

    def set_state(self, state: Sequence[float]):
        a = np.asarray(state)
        self.pos = a[0:2].astype(float)
        self.vel = a[2:4].astype(float)
