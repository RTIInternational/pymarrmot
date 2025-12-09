from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic

def evap_14(p1, p2, Ep, S1, S2, S2min, dt):
    """
    Exponentially scaled evaporation that only activates if another store goes below a certain threshold.
    
    Constraints: f <= S1/dt
    Inputs: 
        p1   - linear scaling parameter [-]
        p2   - exponential scaling parameter [-]
        Ep   - potential evapotranspiration rate [mm/d]
        S1   - current storage in S1 [mm]
        S2   - current storage in S2 [mm]
        S2min- threshold for evaporation deactivation [mm]
        dt   - time step size [d]
    """
    out = min((p1 ** p2) * Ep, S1 / dt) * smooth_threshold_storage_logistic(S2, S2min)
    return out
