# Python function for evap_10
def evap_10(p1, S, Smax, Ep, dt):
    # Flux function
    # Description: Evaporation from bare soil scaled by relative storage
    # Constraints: Ea <= Ep, Ea <= S/dt
    # Inputs: p1 - fraction of area that is bare soil [-]
    #         S - current storage [mm]
    #         Smax - maximum storage [mm]
    #         Ep - potential evapotranspiration rate [mm/d]
    #         dt - time step size [d]
    out = max(min(p1 * S / Smax * Ep, S / dt), 0)
    return out
