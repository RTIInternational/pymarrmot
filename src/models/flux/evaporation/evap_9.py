
def evap_9(S1, S2, p1, Smax, Ep, dt):
    # Flux function
    # Description:  Evaporation from bare soil scaled by relative storage and
    #               by relative water availability across all stores
    # Constraints:  f <= S/dt
    #               f >= 0
    # Inputs:       S1   - current storage in store 1 [mm]
    #               S2   - current storage in store 2 [mm]
    #               p1   - fraction vegetated area [-]
    #               Smax - maximum storage [mm]
    #               Ep   - potential evapotranspiration rate [mm/d]
    #               dt   - time step size [d]

    if S1 + S2 == 0:
        out = 0
    else:
        out = max(min(S1/(S1+S2)*(1-p1)*S1/(Smax-S2)*Ep, S1/dt), 0)

    return out
