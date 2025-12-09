
def interception_3(p1, In):
    """
    Interception excess after a fraction is intercepted
    
    Parameters:
    p1 : float
        Fraction throughfall [-]
    In : float
        Incoming flux [mm/day]
        
    Returns:
    out : float
        Interception excess
    """
    out = p1 * In
    return out
    