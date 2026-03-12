import numpy as np
from typing import List, Dict, Any

def calculate_diversity(positions: List[np.ndarray]) -> float:
    """Calculates the average distance between particles to monitor swarm collapse."""
    if not positions:
        return 0.0
    centroid = np.mean(positions, axis=0)
    distances = [np.linalg.norm(p - centroid) for p in positions]
    return float(np.mean(distances))

def clip_velocity(velocity: np.ndarray, max_val: float = 0.2) -> np.ndarray:
    """Prevents particles from exploding out of the search space."""
    return np.clip(velocity, -max_val, max_val)
