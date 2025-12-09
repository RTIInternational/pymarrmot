
def evap_2(p1, S, Smax, Ep, dt):
    # Flux function
    # Description: Evaporation at a scaled, plant-controlled rate
    # Constraints: f <= Ep
    #              f <= S/dt
    # @(Inputs):   p1   - plant-controlled base evaporation rate [mm/d]
    #             S    - current storage [mm]
    #             Smax - maximum storage [mm]
    #             Ep   - potential evapotranspiration rate [mm/d]
    #             dt   - time step size [d]
    
    return min([p1*S/Smax, Ep, S/dt])
