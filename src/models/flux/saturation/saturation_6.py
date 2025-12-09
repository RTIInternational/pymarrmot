def saturation_6(p1, S, Smax, In):
    # Flux function
    # Description: Saturation excess from a store with different degrees of saturation (linear variant)
    # Constraints: -
    # @(Inputs): p1 - linear scaling parameter [-]
    #            S - current storage [mm]
    #            Smax - maximum contributing storage [mm]
    #            In - incoming flux [mm/d]
    
    out = p1 * S / Smax * In
    return out
