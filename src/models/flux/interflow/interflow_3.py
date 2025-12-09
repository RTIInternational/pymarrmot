
def interflow_3(p1, p2, S, dt):
    # Flux function
    # Description: Non-linear interflow (variant)
    # Constraints: f <= S, S >= 0 (this avoids numerical issues with complex numbers)
    # Inputs: p1 - time delay [d-1]
    #         p2 - exponential scaling parameter [-]
    #         S - current storage [mm]
    #         dt - time step size [d]
    return min(p1*max(S,0)**(p2), max(S/dt, 0))
