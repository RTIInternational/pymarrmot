import numpy as np
from pymarrmot.functions.flux_smoothing.smooth_threshold_temperature_logistic import smooth_threshold_temperature_logistic as sttl

def snowfall_1(In: float, T: float, p1: float, *args: float) -> float:
    """
    Calculates snowfall based on a temperature threshold.

    Parameters
    ----------
    In : float
        Incoming precipitation flux [mm/d].
    T : float
        Current temperature [°C].
    p1 : float
        Temperature threshold below which snowfall occurs [°C].
    args : float, optional
        Smoothing variable r (default is 0.01).

    Returns
    -------
    float
        Outgoing snowfall flux [mm/d].
    """
    if len(args) == 0:
        r = 0.01
    elif len(args) == 1:
        r = args[0]
    else:
        raise ValueError("Too many input arguments for smoothing variable.")

    sttl_result = sttl(T, p1, r)
    out = In * sttl_result
    return out