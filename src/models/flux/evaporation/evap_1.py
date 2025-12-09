def evap_1(S, Ep, dt):
    # Flux function
    # Description: Evaporation at the potential rate
    # Constraints: f <= S/dt
    # @(Inputs): S    - current storage [mm]
    #            Ep   - potential evaporation rate [mm/d]
    #            dt   - time step size
    out = min(S/dt, Ep)
    return out