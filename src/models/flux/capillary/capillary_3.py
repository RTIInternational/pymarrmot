from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic

def capillary_3(p1, p2, S1, S2, dt):
    # Flux function
    # Description: Capillary rise scaled by receiving store's deficit up to a storage threshold
    # Constraints: f <= S2
    # Inputs: p1   - base capillary rise rate [mm/d]
    #         p2   - threshold above which no capillary flow occurs [mm]
    #         S1   - current storage in receiving store [mm]
    #         S2   - current storage in supplying store [mm]
    #         dt   - time step size [d]
    return min(S2/dt, p1*(1-S1/p2)*smooth_threshold_storage_logistic(S1, p2))
