import numpy as np

def evap_3(p1: float, S: float, Smax: float, Ep: float, dt: float) -> float:
    """
    Evaporation based on scaled current water storage and wilting point.

    Parameters
    ----------
    p1 : float
        Wilting point as fraction of Smax [-].
    S : float
        Current storage [mm].
    Smax : float
        Maximum storage [mm].
    Ep : float
        Potential evapotranspiration rate [mm/d].
    dt : float
        Time step size [d].

    Returns
    -------
    float
        Evaporation rate [mm/d].
    """
    
    out = min(S / (p1 * Smax) * Ep, Ep, S / dt)
    
    return out
