import numpy as np
from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic as stsl

def area_1(p1: float, p2: float, S: float, Smin: float, Smax: float, *args) -> float:
    """
    Auxiliary function that calculates a variable contributing area.

    Parameters
    ----------
    p1 : float
        Linear scaling parameter [-].
    p2 : float
        Exponential scaling parameter [-].
    S : float
        Current storage [mm].
    Smin : float
        Minimum contributing storage [mm].
    Smax : float
        Maximum contributing storage [mm].
    args : tuple, optional
        Additional arguments for smoothing. Default is (0.01, 5.00).

    Returns
    -------
    float
        Variable contributing area.
    """
    
    if len(args) == 0:
        smoothing_factor = stsl(S, Smin)
    elif len(args) == 1:
        smoothing_factor = stsl(S, Smin, args[0])
    elif len(args) == 2:
        smoothing_factor = stsl(S, Smin, args[0], args[1])
    
    area = np.minimum(1, p1 * (np.maximum(0, S - Smin) / (Smax - Smin)) ** p2) * (1 - smoothing_factor)
    
    return area