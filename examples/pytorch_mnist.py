import torch
import torch.nn as nn
import torch.optim as optim
from swarm_opt import SwarmOptimizer, SearchSpace

def train_and_evaluate(params):
    # Mock training function for demonstration
    lr = params['lr']
    batch_size = params['batch_size']
    hidden_size = params['hidden_size']
    
    # Simulate a score based on some "optimal" values
    # In reality, this would be your training loop returning validation accuracy
    score = -((lr - 0.01)**2) - ((hidden_size - 128)/512)**2
    return float(score)

if __name__ == "__main__":
    space_config = {
        'lr': (1e-4, 1e-1, 'log'),
        'batch_size': (16, 256, 'int'),
        'hidden_size': (32, 512, 'int')
    }
    
    space = SearchSpace(space_config)
    optimizer = SwarmOptimizer(
        search_space=space,
        objective_func=train_and_evaluate,
        n_particles=15,
        max_iter=20
    )
    
    best_params = optimizer.optimize()
    print(f"Best Parameters Found: {best_params}")
