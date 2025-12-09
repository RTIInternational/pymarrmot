
def saturation_8(p1, p2, S, Smax, In):
    '''
    saturation_8

    Flux function
    Description: Saturation excess flow from a store with different degrees of saturation (min-max linear variant)
    Constraints: -
    Inputs: p1 - minimum fraction contributing area [-]
            p2 - maximum fraction contributing area [-]
            S - current storage [mm]
            Smax - maximum contributing storage [mm]
            In - incoming flux [mm/d]
    '''
    out = (p1 + (p2 - p1) * S / Smax) * In
    return out
