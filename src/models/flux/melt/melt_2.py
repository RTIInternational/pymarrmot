
def melt_2(p1, S, dt):
    """
    Snowmelt at a constant rate
    Constraints: f <= S/dt
    Inputs: p1 - melt rate [mm/d]
            S - current storage [mm]
            dt - time step size [d]
    """
    return min(p1, S/dt)
