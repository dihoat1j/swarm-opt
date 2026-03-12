import unittest
import numpy as np
from swarm_opt.space import SearchSpace
from swarm_opt.optimizer import SwarmOptimizer

class TestSwarm(unittest.TestCase):
    def test_search_space_mapping(self):
        config = {'a': (0, 10, 'float')}
        space = SearchSpace(config)
        params = space.inverse_transform(np.array([0.5]))
        self.assertAlmostEqual(params['a'], 5.0)

    def test_optimizer_convergence(self):
        space = SearchSpace({'x': (-10, 10, 'float')})
        # Objective: maximize -(x-2)^2
        def obj(p): return -(p['x'] - 2)**2
        
        opt = SwarmOptimizer(space, obj, n_particles=10, max_iter=10)
        best = opt.optimize()
        self.assertAlmostEqual(best['x'], 2.0, delta=0.5)

if __name__ == '__main__':
    unittest.main()
