
def evap_5(p1, S, Smax, Ep, dt):
    '''
    evap_5 evaporation based on scaled current water storage, for a fraction
    of the surface

    Inputs:
    - p1   : fraction of area that is bare soil [-]
    - S    : current storage [mm]
    - Smax : maximum storage [mm]
    - Ep   : potential evapotranspiration rate [mm/d]
    - dt   : time step size [d]
    
    Output:
    - out  : evaporation value
    '''
    import numpy as np
    
    # Evaporation calculation
    out = np.maximum(np.minimum((1-p1)*(S/Smax)*Ep, S/dt), 0)
    
    return out
