from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic

def baseflow_7(p1, p2, S, dt):
    """
    Flux function.

    Description: Non-linear outflow from a reservoir.

    Constraints: f <= S / dt
                 S >= 0

    Parameters:
    -----------
    p1 : float
        Time coefficient [d^-1].
    p2 : float
        Exponential scaling parameter [-].
    S : float
        Current storage [mm].
    dt : float
        Time step size [d].

    Returns:
    --------
    out : float
        Resulting flux.
    """
    out = min(S / dt, p1 * max(0, S) ** p2)
    return out