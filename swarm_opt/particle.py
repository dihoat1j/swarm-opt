import numpy as np
from typing import Dict, Any

class Particle:
    """
    Represents a single agent in the swarm.
    """
    def __init__(self, search_space):
        self.search_space = search_space
        self.dim = search_space.dimensions
        self.position = np.random.uniform(0, 1, self.dim)
        self.velocity = np.random.uniform(-0.1, 0.1, self.dim)
        self.best_pos = self.position.copy()
        self.best_score = float('-inf')

    def get_params(self) -> Dict[str, Any]:
        return self.search_space.inverse_transform(self.position)

    def update_best(self, score: float):
        if score > self.best_score:
            self.best_score = score
            self.best_pos = self.position.copy()

    def step(self, g_best: np.ndarray, w: float, c1: float, c2: float):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        
        # Velocity update
        cognitive = c1 * r1 * (self.best_pos - self.position)
        social = c2 * r2 * (g_best - self.position)
        self.velocity = w * self.velocity + cognitive + social
        
        # Position update
        self.position = np.clip(self.position + self.velocity, 0, 1)
