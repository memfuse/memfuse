"""Serialization utilities for MemFuse server."""

import numpy as np
from typing import Any


def convert_numpy_types(obj: Any) -> Any:
    """Convert NumPy types to Python native types.

    Args:
        obj: Object to convert

    Returns:
        Converted object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def prepare_response_data(data: Any) -> Any:
    """Prepare data for API response by converting NumPy types.

    Args:
        data: Data to prepare

    Returns:
        Prepared data
    """
    return convert_numpy_types(data)
