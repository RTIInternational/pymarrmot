
def infiltration_6(p1, p2, S, Smax, fin):
    """
    Creates scaled, non-linear infiltration
    """

    # Infiltration rate non-linearly scaled by relative storage
    # Constraints:  f <= fin
    # @(Inputs):    p1   - base infiltration rate [mm/d]
    #               p2   - exponential scaling parameter [-]
    #               S    - current storage [mm]
    #               Smax - maximum contributing storage [mm]
    #               fin  - incoming flux [mm/d]

    out = min([fin, p1 * max(0, S / Smax) ** p2 * fin])
    return out
