from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic

def interflow_11(p1, p2, S, dt, *varargin):
    """
    Flux function.
    
    Description: Constant interflow if storage exceeds a threshold.
    
    Constraints: f <= (S - p2) / dt
    
    Parameters:
    -----------
    p1 : float
        Interflow rate [mm/d].
    p2 : float
        Storage threshold for flow generation [mm].
    S : float
        Current storage [mm].
    dt : float
        Time step size [d].
    varargin : tuple
        Smoothing variables r (default 0.01) and e (default 5.00).
    
    Returns:
    --------
    out : float
        Resulting flux.
    """
    if len(varargin) == 0:
        out = min(p1, (S - p2) / dt) * (1 - smooth_threshold_storage_logistic(S, p2))
    elif len(varargin) == 1:
        out = min(p1, (S - p2) / dt) * (1 - smooth_threshold_storage_logistic(S, p2, varargin[0]))
    elif len(varargin) == 2:
        out = min(p1, (S - p2) / dt) * (1 - smooth_threshold_storage_logistic(S, p2, varargin[0], varargin[1])) 
    
    return out