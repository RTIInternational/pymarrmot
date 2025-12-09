import numpy as np

def interflow_5(p1: float, S: float) -> float:
    """
    Calculates linear interflow.

    Parameters
    ----------
    p1 : float
        Time coefficient [d-1].
    S : float
        Current storage [mm].

    Returns
    -------
    float
        Linear interflow value.
    """
    out = p1 * S
    return out
