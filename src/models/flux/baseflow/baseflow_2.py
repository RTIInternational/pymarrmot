
def baseflow_2(S, p1, p2, dt):
    '''
    Flux function
    Description: Non-linear outflow from a reservoir
    Constraints: f <= S/dt, S >= 0 (prevents numerical issues with complex numbers)
    (Inputs): S - current storage [mm]
             p1 - time coefficient [d]
             p2 - exponential scaling parameter [-]
             dt - time step size [d]
    '''
    out = min((1./p1 * max(S,0))**(1./p2), max(S,0)/dt)
    return out
