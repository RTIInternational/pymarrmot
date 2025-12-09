# Python code
def percolation_2(p1, S, Smax, dt):
    # Percolation scaled by current relative storage
    # Constraints: f <= S/dt
    # Inputs: p1 - maximum percolation rate [mm/d]
    #         S - current storage [mm]
    #         Smax - maximum storage [mm]
    #         dt - time step size [d]
    out = min(S/dt, p1 * S / Smax)
    return out
