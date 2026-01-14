import numpy as np
import random
import os
import sys
import optuna

def setup_project_paths():
    """
    Set up sys.path and change working directory to project root.
    
    This function ensures that:
    1. Python can import modules from src/ (via sys.path)
    2. Relative paths in config files work correctly (via os.chdir)
    
    Should be called at the beginning of notebook scripts before importing
    any project modules or configs.
    
    Returns:
        str: The project root directory path
    """
    # Get the project root by going up 3 levels from this file
    # (src/common/utils.py -> src/common -> src -> project root)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Add project root to sys.path for imports
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Change working directory to project root so relative paths work
    os.chdir(project_root)

    return project_root


  
def set_global_seeds(seed):
    """
    Set random seeds for reproducibility across multiple libraries.
    
    This function sets consistent random seeds for numpy, Python's random module,
    and Python's hash randomization to ensure reproducible results.
    
    Args:
        seed (int): The random seed to use for all random number generators
    
    Returns:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

