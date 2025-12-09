import numpy as np

def snowfall_2(In: float, T: float, p1: float, p2: float) -> float:
    """
    Calculates snowfall based on a temperature threshold interval.

    Parameters
    ----------
    In : float
        Incoming precipitation flux [mm/d].
    T : float
        Current temperature [°C].
    p1 : float
        Midpoint of the combined rain/snow interval [°C].
    p2 : float
        Length of the mixed snow/rain interval [°C].

    Returns
    -------
    float
        Snowfall amount [mm/d].
    """
    out = min(In, max(0, In * (p1 + 0.5 * p2 - T) / p2))
    return out
