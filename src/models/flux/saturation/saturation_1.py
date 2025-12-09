from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic

def saturation_1(In, S, Smax, *varargin):
    """
    SATURATION_1: Flux function for saturation excess from a store that has reached maximum capacity

    Inputs:
    - In: incoming flux [mm/d]
    - S: current storage [mm]
    - Smax: maximum storage [mm]
    - varargin(1): smoothing variable r (default 0.01)
    - varargin(2): smoothing variable e (default 5.00)

    Output:
    - out: calculated output
    """
    if len(varargin) == 0:
        out = In * (1 - smooth_threshold_storage_logistic(S, Smax))
    elif len(varargin) == 1:
        out = In * (1 - smooth_threshold_storage_logistic(S, Smax, varargin[0]))
    elif len(varargin) == 2:
        out = In * (1 - smooth_threshold_storage_logistic(S, Smax, varargin[0], varargin[1]))
    
    return out
