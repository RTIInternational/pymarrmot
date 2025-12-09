
def refreeze_1(p1, p2, p3, T, S, dt):
    '''
    Flux function
    Description: Refreezing of stored melted snow
    Constraints: f <= S/dt
    Inputs:
      p1 - reduction fraction of degree-day-factor [-]
      p2 - degree-day-factor [mm/oC/d]
      p3 - temperature threshold for snowmelt [oC]
      T  - current temperature [oC]
      S  - current storage [mm]
      dt - time step size [d]
    '''

    out = max(min(p1*p2*(p3-T), S/dt), 0)
    return out
