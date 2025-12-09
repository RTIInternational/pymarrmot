
def capillary_1(p1, S1, S1max, S2, dt):
    # Flux function
    # ------------------
    # Description:  Capillary rise: based on deficit in higher reservoir
    # Constraints:  f <= S2/dt
    # @(Inputs):    p1   - maximum capillary rise rate  [mm/d]
    #               S1   - current storage in receiving store [mm]
    #               S1max- maximum storage in receiving store [mm]
    #               S2   - current storage in providing store [mm]
    #               dt   - time step size [d]
    
    out = min(p1 * (1 - S1 / S1max), S2 / dt)
    return out
