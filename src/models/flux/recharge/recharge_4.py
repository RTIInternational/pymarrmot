
def recharge_4(p1, S, dt):
    # Flux function
    # Description:  Constant recharge
    # Constraints:  f <= S/dt
    # @(Inputs):    p1   - time coefficient [d-1]
    #               S    - current storage [mm]
    #               dt   - time step size [d]
    out = min(p1, S/dt)
    return out
