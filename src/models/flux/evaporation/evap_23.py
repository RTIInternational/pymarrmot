import numpy as np

def evap_23(p1: float, p2: float, S: float, Smax: float, Ep: float, dt: float) -> float:
    """
    Combines evaporation and transpiration processes.

    Transpiration from vegetation at the potential rate if storage is 
    above field capacity and scaled by relative storage if not, 
    addition of evaporation from bare soil scaled by relative storage.

    Parameters
    ----------
    p1 : float
        Fraction vegetated area [-] (0...1).
    p2 : float
        Field capacity coefficient [-].
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
        The combined evaporation and transpiration rate [mm/d].
    """
    term1 = p1 * Ep + (1 - p1) * S / Smax * Ep
    term2 = p1 * Ep * S / (p2 * Smax) + (1 - p1) * S / Smax * Ep
    term3 = S / dt
    out = min(term1, term2, term3)
    return out