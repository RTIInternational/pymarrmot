
def excess_1(So, Smax, dt):
    # Flux function
    # ------------------
    # Description:  Storage excess when store size changes (returns flux [mm/d])
    # Constraints:  f >= 0
    # @(Inputs):    So   - 'old' storage [mm]
    #               Smax - 'new' maximum storage [mm]
    #               dt   - time step size [d]
    
    out = max((So - Smax) / dt, 0)
    return out
