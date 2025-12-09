
def evap_20(p1, p2, S, Smax, Ep, dt):
    '''
    Evaporation limited by a maximum evaporation rate and scaled below a wilting point
    :param p1: maximum evaporation rate [mm/d]
    :param p2: wilting point as fraction of Smax [-]
    :param S: current storage [mm]
    :param Smax: maximum storage [mm]
    :param Ep: potential evapotranspiration rate [mm/d]
    :param dt: time step size [d]
    :return: evaporation flux
    '''
    return min([p1 * S / (p2 * Smax), Ep, S / dt])
