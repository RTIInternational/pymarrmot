import numpy as np

def saturation_10(p1: float, p2: float, p3: float, S: float, In: float) -> float:
    """
    Calculate the saturation excess flow from a store with different degrees of saturation (min-max exponential variant).

    Parameters
    ----------
    p1 : float
        Maximum contributing fraction area [-].
    p2 : float
        Minimum contributing fraction area [-].
    p3 : float
        Exponential scaling parameter [-].
    S : float
        Current storage [mm].
    In : float
        Incoming flux [mm/d].

    Returns
    -------
    float
        The saturation excess flow [mm/d].
    """
    # added if statement to prevent overflow error
    if (p3 * S) > 700:
        out = p1 * In
    else:    
        out = min(p1, p2 + p2 * np.exp(p3 * S)) * In
    return out