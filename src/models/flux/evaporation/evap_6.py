
def evap_6(p1, p2, S, Smax, Ep, dt):
    '''
    evap_6 evaporation based on scaled current water storage, a wilting point, a constraining factor and limited by potential rate.
    
    Original MATLAB function description:
    # Flux function
    # ------------------
    # Description:  Transpiration from vegetation at the potential rate if 
    #               storage is above a wilting point and scaled by relative 
    #               storage if not
    # Constraints:  Ea <= Ep
    #               Ea <= S/dt
    # @(Inputs):    p1   - fraction vegetated area [-]
    #               p2   - wilting point as fraction of Smax
    #               S    - current storage [mm]
    #               Smax - maximum storage [mm]
    #               Ep   - potential evapotranspiration rate [mm/d]
    #               dt   - time step size [d]
    '''
    return min([p1*Ep, p1*Ep*S/(p2*Smax), S/dt])
