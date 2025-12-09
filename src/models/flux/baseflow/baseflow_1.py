
import numpy as np

def baseflow_1(p1: float, S: float) -> float:
    """
    Calculate the outflow from a linear reservoir.

    Parameters
    ----------
    p1 : float
        Time scale parameter [d-1].
    S : float
        Current storage [mm].

    Returns
    -------
    float
        The calculated baseflow outflow.
    """
    out = p1 * S
    return out
