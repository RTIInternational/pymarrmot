
def capillary_2(p1, S, dt):
    # Capillary rise at constant rate
    # Constraints: f <= S/dt
    # Inputs: p1 - base capillary rise rate [mm/d]
    #         S - current storage [mm]
    #         dt - time step size [d]
    
    return min(p1, S/dt)
