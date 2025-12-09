from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic

def melt_3(p1, p2, T, S1, S2, St, dt, *varargin):
    """
    Flux function
    Description:  Glacier melt provided no snow is stored on the ice layer
    Constraints:  f <= S1/dt
    
    Inputs:
        p1: degree-day factor [mm/oC/d]
        p2: temperature threshold for snowmelt [oC]
        T: current temperature [oC]
        S1: current storage in glacier [mm]
        S2: current storage in snowpack [mm]
        St: storage in S2 threshold below which glacier melt occurs [mm]
        dt: time step size [d]
        varargin(1): smoothing variable r (default 0.01)
        varargin(2): smoothing variable e (default 5.00)
    
    Returns:
        out: calculated glacier melt value
    """
    if len(varargin) == 0:
        out = min(max(p1*(T-p2),0),S1/dt)*smooth_threshold_storage_logistic(S2, St)
    elif len(varargin) == 1:
        out = min(max(p1*(T-p2),0),S1/dt)*smooth_threshold_storage_logistic(S2, St, varargin[0])
    elif len(varargin) == 2:
        out = min(max(p1*(T-p2),0),S1/dt)*smooth_threshold_storage_logistic(S2, St, varargin[0], varargin[1])
    return out
