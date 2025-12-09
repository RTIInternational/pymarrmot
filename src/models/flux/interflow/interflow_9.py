
def interflow_9(S, p1, p2, p3, dt):
    # Non-linear interflow if storage exceeds a threshold
    # Constraints:  f <= S-p2
    #               S-p2 >= 0     prevents numerical issues with complex numbers
    # @(Inputs):    p1   - time coefficient [d-1]
    #               p2   - storage threshold for flow generation [mm]
    #               p3   - exponential scaling parameter [-]   
    #               S    - current storage [mm]
    #               dt   - time step size [d]
    
    out = min(max((S - p2) / dt, 0), (p1 * max(S - p2, 0))**p3)
    return out
