
def baseflow_5(p1, p2, S, Smax, dt):
    """
    Non-linear scaled outflow from a reservoir.
    
    Constraints: f <= S/dt
    Inputs:
        p1: base outflow rate [mm/d]
        p2: exponential scaling parameter [-]
        S: current storage [mm]
        Smax: maximum contributing storage [mm]
        dt: time step size [d]
    Output:
        out: scaled outflow from the reservoir
    """
    out = min(S/dt, p1 * ((max(S, 0) / Smax) ** p2))
    return out
