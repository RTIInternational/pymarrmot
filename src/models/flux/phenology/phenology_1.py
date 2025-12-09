import numpy as np

def phenology_1(T: float, p1: float, p2: float, Ep: float) -> float:
    """
    Corrects potential evapotranspiration (Ep) for phenology effects.

    Parameters
    ----------
    T : float
        Current temperature [°C].
    p1 : float
        Temperature threshold where evaporation stops [°C].
    p2 : float
        Temperature threshold above which corrected Ep equals Ep.
    Ep : float
        Current potential evapotranspiration [mm/d].

    Returns
    -------
    float
        Corrected potential evapotranspiration [mm/d].
    """
    correction_factor = np.clip((T - p1) / (p2 - p1), 0, 1)
    out = correction_factor * Ep
    return out
