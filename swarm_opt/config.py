import logging
from swarm_opt.optimizer import SwarmOptimizer
from swarm_opt.space import SearchSpace

# Configure logging for the library
logger = logging.getLogger('swarm_opt')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
