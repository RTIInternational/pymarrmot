
def evap_8(S1, S2, p1, p2, Ep, dt):
    # evap_8 evaporation based on scaled current water storage, a wilting point, and a distribution factor.
    
    # Flux function
    # ------------------
    # Description:  Transpiration from vegetation, at potential rate if soil 
    #               moisture is above the wilting point, and linearly 
    #               decreasing if not. Also scaled by relative storage across 
    #               all stores
    # Constraints:  f <= S/dt
    #               f >= 0
    # @(Inputs):    S1   - current storage in store 1 [mm]
    #               S2   - current storage in store 2 [mm]
    #               p1   - fraction vegetated area [-]
    #               p2   - wilting point [mm]
    #               Ep   - potential evapotranspiration rate [mm/d]
    #               dt   - time step size [d]

    if S1 + S2 == 0:
        out = 0
    else:
        out = max(min([S1/(S1+S2)*p1*Ep, S1/(S1+S2)*S1/p2*p1*Ep, S1/dt]),0)
    
    return out
