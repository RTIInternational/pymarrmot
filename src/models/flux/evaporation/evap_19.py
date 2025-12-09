
def evap_19(p1, p2, S, Smax, Ep, dt):
    # Description: Non-linear scaled evaporation
    # Constraints: f <= Ep, f <= S/dt
    # Inputs: p1 - linear scaling parameter [-]
    #         p2 - exponential scaling parameter [-]
    #         S - current storage [mm]
    #         Smax - maximum storage [mm]
    #         Ep - potential evapotranspiration rate [mm/d]
    #         dt - time step size [d]
    out = min([S/dt, Ep, p1 * max(0, S/Smax)**(p2) * Ep])
    return out
