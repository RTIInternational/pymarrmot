
def evap_17(p1, S, Ep):
    '''
    Flux function
    Description: Scaled evaporation from a store that allows negative values
    Constraints: -
    Inputs: 
        p1 - linear scaling parameter [mm-1]
        S - current storage [mm]
        Ep - potential evapotranspiration rate [mm/d]
    '''
    import numpy as np
    
    #bug fix: 11July2024 - SAS - if else statement added to prevent overflow in np.exp term
    if abs(-1 * p1 * S) > 700:
        out = 0
    else:
        out = 1 / (1 + np.exp(-1 * p1 * S)) * Ep
    
    return out
