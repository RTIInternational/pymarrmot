
def split_2(p1: float, In: float) -> float:
    """
    Split the flow.

    Parameters
    ----------
    p1 : float
        Fraction of flux to be diverted [-].
    In : float
        Incoming flux [mm/d].

    Returns
    -------
    float
        The split flow [mm/d].
    """
    out = (1-p1) * In
    return out
