
def infiltration_8(S, Smax, fin):
    # Infiltration into storage is equal to the inflow when current storage is under the maximum storage,
    # and zero when storage reaches maximum capacity
    # Constraints:  f <= fin
    # @(Inputs): 
    #     S    - current storage [mm]
    #     Smax - maximum storage [mm]
    #     fin  - size of incoming flux [mm/d]

    out = (S < Smax) * fin
    return out
