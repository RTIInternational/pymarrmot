import numpy as np
from typing import Tuple, Optional
from pymarrmot.functions.flux_smoothing.smooth_threshold_temperature_logistic import smooth_threshold_temperature_logistic

def rainfall_1(In: np.ndarray, T: np.ndarray, p1: float, r: Optional[float] = 0.01) -> np.ndarray:
    """
    Rainfall based on temperature threshold.

    Parameters
    ----------
    In : np.ndarray
        Incoming precipitation flux [mm/d].
    T : np.ndarray
        Current temperature [°C].
    p1 : float
        Temperature threshold above which rainfall occurs [°C].
    r : Optional[float], default=0.01
        Smoothing variable.

    Returns
    -------
    np.ndarray
        Modified precipitation flux [mm/d].
    """
    
    out = In * (1 - smooth_threshold_temperature_logistic(T, p1, r))
    return out