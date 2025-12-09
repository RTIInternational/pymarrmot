
def interflow_4(p1, p2, S):
    # Flux function
    # Description: Combined linear and scaled quadratic interflow
    # Constraints: f <= S
    #              S >= 0 - prevents numerical issues with complex numbers
    # @(Inputs):   p1   - time coefficient [d-1]
    #              p2   - scaling factor [mm-1 d-1]
    #              S    - current storage [mm]
    
    return min(max(S, 0), p1 * max(S, 0) + p2 * max(S, 0) ** 2)
