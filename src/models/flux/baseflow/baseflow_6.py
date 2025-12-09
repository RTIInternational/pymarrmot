from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic

def baseflow_6(p1, p2, S, dt):
    # Description: Quadratic outflow from a reservoir if a storage threshold is exceeded
    # Constraints: f <= S/dt
    # Inputs: p1 - linear scaling parameter [mm-1 d-1]
    #         p2 - threshold that must be exceeded for flow to occur [mm]
    #         S - current storage [mm]
    out = min(S / dt, p1 * S**2) * (1 - smooth_threshold_storage_logistic(S, p2))
    return out
