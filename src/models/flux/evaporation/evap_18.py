
import numpy as np

def evap_18(p1: float, p2: float, p3: float, S: float, Ep: float) -> float:
    """
    Calculate the exponentially declining evaporation from the deficit store.

    Parameters
    ----------
    p1 : float
        Linear scaling parameter [-].
    p2 : float
        Linear scaling parameter [-].
    p3 : float
        Storage scaling parameter [mm].
    S : float
        Current storage [mm].
    Ep : float
        Potential evapotranspiration rate [mm/d].

    Returns
    -------
    float
        The evaporation rate [mm/d].
    """
    out = p1 * np.exp(-1.0 * p2 * S / p3) * Ep
    return out