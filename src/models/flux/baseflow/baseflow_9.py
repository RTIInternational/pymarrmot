
def baseflow_9(p1, p2, S):
    """
    Linear flow above a threshold
    
    Arguments:
    p1 : float - time coefficient [d-1]
    p2 : float - storage threshold for flow generation [mm]
    S : float - current storage [mm]
    
    Returns:
    out : float - resulting flow
    """
    out = p1 * max(0, S - p2)
    return out
