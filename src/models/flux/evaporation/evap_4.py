def evap_4(Ep, p1, S, p2, Smax, dt):
    """
    Evaporation based on scaled current water storage, a wilting point,
    a constraining factor and limited by potential rate.
    
    Args:
    - Ep: potential evapotranspiration rate [mm/d]
    - p1: scaling parameter [-]
    - p2: wilting point as fraction of Smax [-]
    - S: current storage [mm]
    - Smax: maximum storage [mm]
    - dt: time step size [d]
    
    Returns:
    - out: calculated evaporation
    """
    out = min(Ep * max(0, p1 * (S - p2 * Smax) / (Smax - p2 * Smax)), S / dt)
    return out
