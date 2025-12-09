
def saturation_3(S, Smax, p1, In):
    """
    Flux function: Saturation excess from a store with different degrees of saturation (exponential variant)
    
    Inputs:
    - S: current storage [mm]
    - Smax: maximum contributing storage [mm]
    - p1: linear scaling parameter [-]
    - In: incoming flux [mm/d]
    
    Returns:
    - out: calculated flux
    """
    import numpy as np  # Import numpy for exponential function
    out = (1 - (1 / (1 + np.exp((S / Smax + 0.5) / p1)))) * In
    return out
