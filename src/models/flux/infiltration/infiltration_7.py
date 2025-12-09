
import numpy as np
from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic as stsl

def infiltration_7(p1: float, p2: float, S: float, Smax: float, In: float, *args) -> np.ndarray:
    """
    Calculates infiltration as an exponentially declining function based on relative storage with customization.

    Parameters
    ----------
    p1 : float
        Maximum infiltration rate [mm/d].
    p2 : float
        Exponential scaling parameter [-].
    S : float
        Current storage [mm].
    Smax : float
        Maximum storage [mm].
    In : float
        Size of incoming flux [mm/d].
    *args : float
        Additional optional parameters for the smoothing function.

    Returns
    -------
    float
        The infiltration flux, constrained to be less than or equal to the incoming flux and customized with smoothing.
    """
    pre_smoother = np.minimum(p1 * np.exp((-1 * p2 * S) / Smax), In)
    
    if len(args) == 0:
        out = pre_smoother * (1 - stsl(S, Smax))
    elif len(args) == 1:
        out = pre_smoother * (1 - stsl(S, Smax, args[0]))
    elif len(args) == 2:
        out = pre_smoother * (1 - stsl(S, Smax, args[0], args[1]))
    else:
        raise ValueError("Too many arguments provided for the smoothing function.")
    
    return out