
import numpy as np

def evap_12(S, p1, Ep):
    # Evaporation from deficit store, with exponential decline as deficit goes below a threshold
    # Inputs:
    #   S  - current storage [mm]
    #   p1 - wilting point [mm]
    #   Ep - potential evapotranspiration rate [mm/d]
    # Returns:
    #   Evaporation rate [mm/d]
    return min(1, np.exp(2*(1-S/p1))) * Ep
