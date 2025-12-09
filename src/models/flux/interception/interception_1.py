
import numpy as np
from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic as stsl

def interception_1(In: np.ndarray, S: np.ndarray, Smax: float, *args: float) -> np.ndarray:
    """
    Interception excess when maximum capacity is reached.

    Parameters
    ----------
    In : np.ndarray
        Incoming flux [mm/d].
    S : np.ndarray
        Current storage [mm].
    Smax : float
        Maximum storage [mm].
    *args : float, optional
        Additional smoothing variables r and e.

    Returns
    -------
    np.ndarray
        Outgoing flux after interception [mm/d].
    """
    if len(args) == 0:
        out = In * (1 - stsl(S, Smax))
    elif len(args) == 1:
        out = In * (1 - stsl(S, Smax, args[0]))
    elif len(args) == 2:
        out = In * (1 - stsl(S, Smax, args[0], args[1]))
    else:
        raise ValueError("Too many arguments provided for smoothing variables.")

    return out
