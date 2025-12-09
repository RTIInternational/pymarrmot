
def evap_7(S, Smax, Ep, dt):
    # evaporation based on scaled current water storage, limited by potential rate.
    
    # Flux function
    # Description:  Evaporation scaled by relative storage
    # Constraints:  f <= S/dt
    # Inputs:    S    - current storage [mm]
    #            Smax - maximum contributing storage [mm]
    #            Ep   - potential evapotranspiration rate [mm/d]
    #            dt   - time step size [d]
    out = min(S/Smax*Ep, S/dt)
    return out
