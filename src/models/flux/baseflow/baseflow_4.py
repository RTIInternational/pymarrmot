
import numpy as np

def baseflow_4(p1: float, p2: float, S: float) -> float:
    """
    Calculates the exponential outflow from a deficit store.

    Parameters
    ----------
    p1 : float
        Base outflow rate [mm/d].
    p2 : float
        Exponent parameter [mm^-1].
    S : np.ndarray
        Current storage [mm].

    Returns
    -------
    float
        The calculated baseflow outflow.
    """
    out = p1 * np.exp(-p2 * S)
    return out