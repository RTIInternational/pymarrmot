
def melt_1(p1, p2, T, S, dt):
    # Description: Snowmelt from degree-day-factor
    # Constraints: f <= S/dt
    # Inputs: p1 - degree-day factor [mm/oC/d]
    #         p2 - temperature threshold for snowmelt [oC]
    #         T - current temperature [oC]
    #         S - current storage [mm]
    #         dt - time step size [d]
    out = max(min(p1 * (T - p2), S/dt), 0)
    return out
