
def recharge_1(p1, S, Smax, flux):
    """
    Recharge as a scaled fraction of the incoming flux.
    
    Arguments:
    p1: float - fraction of flux that is recharge [-]
    S: float - current storage [mm]
    Smax: float - maximum contributing storage [mm]
    flux: float - incoming flux [mm/d]
    
    Returns:
    out: float - recharge amount
    """
    out = p1 * (S / Smax) * flux
    return out
