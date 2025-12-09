
def percolation_5(p1, p2, S, Smax, dt):
    # Flux function
    # Description: Non-linear percolation
    # Constraints: f <= S/dt, S >= 0 (prevents complex numbers)
    # Inputs: p1 - base percolation rate [mm/d]
    #         p2 - exponential scaling parameter [-]
    #         S - current storage [mm]
    #         Smax - maximum contributing storage [mm]
    #         dt - time step size [d]
    
    return min(S/dt, p1 * ((max(S, 0) / Smax) ** p2))
