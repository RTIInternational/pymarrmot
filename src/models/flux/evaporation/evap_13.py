def evap_13(p1, p2, Ep, S, dt):
    # Flux function
    # Description: Exponentially scaled evaporation
    # Constraints: f <= S/dt
    # Inputs: 
    #   p1 - linear scaling parameter [-]
    #   p2 - exponential scaling parameter [-]
    #   Ep - potential evapotranspiration rate [mm/d]
    #   S - current storage [mm]
    #   dt - time step size [d]
    
    out = min((p1**p2)*Ep, S/dt)
    return out
