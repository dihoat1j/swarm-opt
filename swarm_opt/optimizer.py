import numpy as np
import logging
from typing import Dict, Any, List, Callable
from concurrent.futures import ProcessPoolExecutor
from .particle import Particle
from .space import SearchSpace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SwarmOptimizer:
    """
    Main engine for Swarm Intelligence Hyperparameter Optimization.
    """
    def __init__(
        self,
        search_space: SearchSpace,
        objective_func: Callable[[Dict[str, Any]], float],
        n_particles: int = 20,
        max_iter: int = 50,
        inertia: float = 0.5,
        cognitive_coeff: float = 1.5,
        social_coeff: float = 1.5
    ):
        self.search_space = search_space
        self.objective_func = objective_func
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.inertia = inertia
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        
        self.particles = [
            Particle(search_space) for _ in range(n_particles)
        ]
        self.global_best_pos = None
        self.global_best_score = float('-inf')

    def optimize(self) -> Dict[str, Any]:
        """Runs the PSO optimization loop."""
        for i in range(self.max_iter):
            scores = []
            # Evaluate current positions
            for particle in self.particles:
                params = particle.get_params()
                score = self.objective_func(params)
                particle.update_best(score)
                
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_pos = particle.position.copy()
                scores.append(score)
            
            # Update velocities and positions
            for particle in self.particles:
                particle.step(
                    self.global_best_pos,
                    self.inertia,
                    self.cognitive_coeff,
                    self.social_coeff
                )
            
            logger.info(f"Iteration {i+1}/{self.max_iter} - Best Score: {self.global_best_score:.4f}")
            
        return self.search_space.inverse_transform(self.global_best_pos)
