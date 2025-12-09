
def interception_4(p1, p2, t, tmax, In, dt):
    """
    Creates function for effective rainfall through interception.
    
    Parameters:
    p1 (float): mean throughfall fraction [-]
    p2 (float): timing of maximum throughfall fraction [d]
    t (float): current time step [-]
    tmax (float): duration of the seasonal cycle [d]
    In (float): incoming flux [mm/d]
    dt (float): time step size [d]
    
    Returns:
    float: the effective rainfall through interception
    """
    import math
    angle = 2*math.pi*(t*dt - p2) / tmax
    out = max(0, p1 + (1-p1)*math.cos(angle)) * In
    return out
