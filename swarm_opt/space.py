import numpy as np
from typing import Dict, List, Any, Tuple

class SearchSpace:
    """
    Handles mapping between normalized [0, 1] space and actual hyperparameter values.
    """
    def __init__(self, config: Dict[str, Tuple[Any, Any, str]]):
        self.config = config
        self.keys = sorted(config.keys())
        self.dimensions = len(self.keys)

    def inverse_transform(self, vector: np.ndarray) -> Dict[str, Any]:
        params = {}
        for i, key in enumerate(self.keys):
            low, high, ptype = self.config[key]
            val = vector[i]
            
            if ptype == 'int':
                params[key] = int(low + val * (high - low))
            elif ptype == 'float':
                params[key] = low + val * (high - low)
            elif ptype == 'log':
                params[key] = float(np.exp(np.log(low) + val * (np.log(high) - np.log(low))))
        return params
