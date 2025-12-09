from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic

def evap_15(Ep, S1, S1max, S2, S2min, dt):
    """
    Flux function.
    
    Description: Scaled evaporation if another store is below a threshold.
    
    Constraints: f <= S1/dt
    
    Parameters:
    -----------
    Ep : float
        Potential evapotranspiration rate [mm/d].
    S1 : float
        Current storage in S1 [mm].
    S1max : float
        Maximum storage in S1 [mm].
    S2 : float
        Current storage in S2 [mm].
    S2min : float
        Minimum storage in S2 [mm].
    dt : float
        Time step size [d].

    Returns:
    --------
    out : float
        Resulting flux.
    """
    return min((S1 / S1max * Ep) * smooth_threshold_storage_logistic(S2, S2min), S1 / dt)
