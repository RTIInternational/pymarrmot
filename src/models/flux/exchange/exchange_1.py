import numpy as np

def exchange_1(p1: float, p2: float, p3: float, S: float, fmax: float, dt: float) -> float:
    """
    Water exchange between aquifer and channel.

    Parameters
    ----------
    p1 : float
        Linear scaling parameter [-].
    p2 : float
        Linear scaling parameter [-].
    p3 : float
        Exponential scaling parameter [-].
    S : float
        Current storage [mm].
    fmax : float
        Maximum flux size [mm/d].
    dt : float
        Time step size [d].

    Returns
    -------
    float
        The calculated water exchange flux.
    """
    out = max((p1 * abs(S / dt) + p2 * (1 - np.exp(-p3 * abs(S / dt)))) * np.sign(S), -fmax)
    return out
