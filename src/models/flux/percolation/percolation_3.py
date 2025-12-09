
def percolation_3(S, Smax):
    """
    Non-linear percolation (empirical) function
    """
    out = Smax**(-4) / 4 * (4/9)**4 * S**5
    return out
