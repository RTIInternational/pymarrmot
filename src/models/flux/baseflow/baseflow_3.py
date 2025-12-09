
def baseflow_3(S, Smax):
    """
    Empirical non-linear outflow from a reservoir
    S: current storage [mm]
    Smax: maximum contributing storage [mm]
    """
    out = Smax**(-4) / 4 * (S**5)
    return out
