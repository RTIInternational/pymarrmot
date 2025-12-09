
def interflow_8(S, p1, p2):
    # Flux function
    # Description: Linear interflow if storage exceeds a threshold
    # Constraints: f = 0 for S < p2
    # Inputs: S - current storage [mm]
    #         p1 - time coefficient [d-1]
    #         p2 - storage threshold before flow occurs [mm]
    
    out = max(0, p1 * (S - p2))
    
    return out
