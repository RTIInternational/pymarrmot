
def evap_22(p1, p2, S, Ep, dt):
    # evap_22 3-part piece-wise evaporation
    # Flux function
    # Description: Threshold-based evaporation rate
    # Constraints: f <= S/dt
    # @(Inputs): p1 - wilting point [mm]
    #            p2 - 2nd (lower) threshold [mm]
    #            S - current storage [mm]
    #            Ep - potential evapotranspiration rate [mm/d]
    #            dt - time step size [d]
    
    return min(S/dt, max(0, min((S-p1)/(p2-p1)*Ep, Ep)))
