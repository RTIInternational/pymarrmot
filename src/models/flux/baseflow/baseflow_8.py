
def baseflow_8(p1, p2, S, Smax):
    '''
    Flux function for exponential scaled outflow from a deficit store
    Constraints: S <= Smax
    Inputs:
        p1: base outflow rate [mm/d]
        p2: exponential scaling parameter [-]
        S: current storage [mm]
        Smax: maximum contributing storage [mm]
    Returns: 
        Computed outflow
    '''
    import numpy as np
    out = p1 * (np.exp(p2 * min(1, max(S, 0) / Smax)) - 1)
    return out
