
def percolation_4(p1, p2, p3, p4, p5, S, Smax, dt):
    """
    Flux function for demand-based percolation scaled by available moisture.
    
    Args:
    p1   : float : base percolation rate [mm/d]
    p2   : float : percolation rate increase due moisture deficiencies [mm/d]
    p3   : float : non-linearity parameter [-]
    p4   : float : summed deficiency across all model stores [mm]
    p5   : float : summed capacity of model stores [mm]
    S    : float : current storage in the supplying store [mm]
    Smax : float : maximum storage in the supplying store [mm]
    dt   : float : time step size [d]
    
    Returns:
    out  : float : percolation rate [mm/d]
    """
    out = max(0, min(S/dt, max(0, S/Smax) * (p1 * (1 + p2 * (p4/p5)**(1 + p3)))))
    return out
