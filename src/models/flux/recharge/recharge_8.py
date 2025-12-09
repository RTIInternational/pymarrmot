
def recharge_8(p1, S, Smax, p2, dt):
    # Flux function
    # Description:  Recharge as non-linear scaling of incoming flux
    # Constraints:  f <= S/dt
    #               S >= 0
    # @(Inputs):    p1   - recharge scaling non-linearity [-]
    #               S    - current storage [mm]
    #               Smax - maximum contributing storage [mm]
    #               p2   - maximum flux rate [mm/d]
    #               dt   - time step size [d]
    
    out = min(p2*((max(S,0)/Smax)**p1),max(S/dt,0))
    
    return out
