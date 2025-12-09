import numpy as np
from scipy.stats import norm

def saturation_13(p1: float, p2: float, S: float, In: float) -> float:
    """
    Calculate the saturation excess flow from a store with different degrees of saturation (normal distribution variant).

    Parameters
    ----------
    p1 : float
        Soil depth where 50% of the catchment contributes to overland flow [mm].
    p2 : float
        Soil depth where 16% of the catchment contributes to overland flow [mm].
    S : float
        Current storage [mm].
    In : float
        Incoming flux [mm/d].

    Returns
    -------
    float
        The saturation excess flow [mm/d].
    """
    S = max(0, S)  # Ensure storage is non-negative
    if S == 0:
        out = 0
    else:
        out = In * norm.cdf(np.log10(S / p1) / np.log10(p1 / p2))
    return out
