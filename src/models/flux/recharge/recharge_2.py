
def recharge_2(p1, S, Smax, flux):
    # Flux function
    # Description: Recharge as a non-linear scaling of incoming flux
    # Constraints: S >= 0
    # @(Inputs): p1 - recharge scaling non-linearity [-]
    #            S - current storage [mm]
    #            Smax - maximum contributing storage [mm]
    #            flux - incoming flux [mm/d]
    
    out = flux * ((max(S, 0) / Smax) ** p1)
    
    return out
