
def evap_11(S, Smax, Ep):
    # Evaporation quadratically related to current soil moisture
    # Constraints:  f >= 0
    # Inputs: S - current storage [mm]
    #         Smax - maximum storage [mm]
    #         Ep - potential evapotranspiration rate [mm/d]
    out = max(0, (2 * S / Smax - (S / Smax) ** 2) * Ep)
    return out
