import numpy as np

def saturation_4(S: float, Smax: float, In: float) -> float:
    ##S: np.ndarray, Smax: np.ndarray, In: np.ndarray) -> np.ndarray:
    """
    Calculates saturation excess from a store with different degrees of saturation (quadratic variant).

    Parameters
    ----------
    S : float
        Current storage [mm].
    Smax : float
        Maximum storage [mm].
    In : float
        Incoming flux [mm/d].

    Returns
    -------
    float
        The saturation excess flux, constrained to be non-negative.
    """
    out = np.maximum(0, (1 - (S / Smax) ** 2) * In)
    return out

# Example usage:
# S = np.array([50, 100])
# Smax = np.array([100, 200])
# In = np.array([10, 20])
# out = saturation_excess_flux(S, Smax, In)
# print(out)
