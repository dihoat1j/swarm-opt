# SwarmOpt

SwarmOpt is a high-performance Swarm Intelligence simulator specifically designed for hyperparameter optimization in Deep Learning. It leverages Particle Swarm Optimization (PSO) to navigate complex, non-convex parameter spaces more efficiently than grid or random search.

## Features

* Decentralized Optimization: Agents explore the search space independently while sharing global bests.
* Multi-Type Support: Handles continuous, integer, and log-scale hyperparameters.
* Framework Agnostic: Works with PyTorch, TensorFlow, JAX, or any Python-based ML library.
* Lightweight: Minimal dependencies, focused on speed and mathematical correctness.

## Installation

```bash
pip install swarm-opt
```

## Quick Start

```python
from swarm_opt import SwarmOptimizer, SearchSpace

# 1. Define your search space
space = SearchSpace({
    'learning_rate': (1e-5, 1e-1, 'log'),
    'dropout': (0.1, 0.5, 'float'),
    'layers': (1, 5, 'int')
})

# 2. Define your objective (e.g., validation accuracy)
def objective(params):
    # model = MyModel(params)
    # return train(model)
    return -(params['learning_rate'] - 0.001)**2 

# 3. Run the swarm
opt = SwarmOptimizer(space, objective, n_particles=20, max_iter=50)
best_hparams = opt.optimize()
print(best_hparams)
```

## Architecture

The system consists of three core components:
1. SearchSpace: Maps the unit hypercube [0, 1]^N to the actual parameter ranges.
2. Particle: Maintains individual velocity and position vectors.
3. SwarmOptimizer: Orchestrates the collective movement and information exchange.

## Contributing

Please see CONTRIBUTING.md for guidelines on how to submit PRs and report issues.

## License

MIT License - see LICENSE for details.
