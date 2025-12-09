
import numpy as np

def infiltration_1(p1: float, p2: float, S: float, Smax: float, fin: float) -> float:
    """
    Calculates infiltration as an exponentially declining function based on relative storage.

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
    fin : float
        Size of incoming flux [mm/d].

    Returns
    -------
    float
        The infiltration flux, constrained to be less than or equal to the incoming flux.
    """
    out = np.minimum(p1 * np.exp((-1 * p2 * S) / Smax), fin)
    return out