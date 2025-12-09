
def percolation_1(p1, S, dt):
    # Flux function
    # Description: Percolation at a constant rate
    # Constraints: f <= S/dt
    # Inputs: p1 - base percolation rate [mm/day]
    #         S  - current storage [mm]
    #         dt - time step size [day]
    out = min(p1, S/dt)
    return out
