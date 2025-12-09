
def recharge_6(p1, p2, S, dt):
    # Flux function
    # Description:  Recharge to fulfil evaporation demand if the receiving 
    #               store is below a threshold
    # Constraints:  f <= S/dt
    #               S >= 0      prevents complex numbers
    # @(Inputs):    p1   - time coefficient [d-1]
    #               p2   - non-linear scaling [mm]
    #               S    - current storage [mm]
    #               dt   - time step size [d]

    out = min(max(S/dt, 0), p1 * max(S, 0)**p2)
    
    return out
