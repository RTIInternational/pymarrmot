from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic

def saturation_9(In, S, St, *varargin):
    """
    Flux function.
    
    Description: Deficit store: Saturation excess from a store that has reached maximum capacity.
    
    Constraints: -
    
    Parameters:
    -----------
    In : float
        Incoming flux [mm/d].
    S : float
        Current storage [mm].
    St : float
        Threshold for flow generation [mm], 0 for deficit store.
    varargin : tuple
        Smoothing variables r (default 0.01) and e (default 5.00).
    
    Returns:
    --------
    out : float
        Resulting flux.
    """
    if len(varargin) == 0:
        out = In * smooth_threshold_storage_logistic(S, St)
    elif len(varargin) == 1:
        out = In * smooth_threshold_storage_logistic(S, St, varargin[0])
    elif len(varargin) == 2:
        out = In * smooth_threshold_storage_logistic(S, St, varargin[0], varargin[1])
    
    return out
