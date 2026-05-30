"""Reproducibility utilities: seed setting for NumPy, Random, and TensorFlow.

Never hardcode seed values. Use get_seed() or set_all_seeds().
"""

from typing import Optional

import numpy as np


def get_seed() -> int:
    """Get the global seed from config.

    Returns:
        int: Seed value (default 42 if config loading fails).
    """
    try:
        from src.utils.config import load_config

        config = load_config("training")
        return int(config.get("seed", 42))
    except (FileNotFoundError, KeyError):
        return 42


def set_all_seeds(seed: Optional[int] = None) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for: numpy, Python random, TensorFlow (if available).

    Args:
        seed: Seed value. If None, reads from config.

    Returns:
        None
    """
    if seed is None:
        seed = get_seed()

    np.random.seed(seed)

    try:
        import random

        random.seed(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass
