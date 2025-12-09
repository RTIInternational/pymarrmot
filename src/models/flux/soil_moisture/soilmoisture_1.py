from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic

def soilmoisture_1(S1, S1max, S2, S2max):
    # Flux function
    # Description: Water rebalance to equal relative storage (2 stores)
    # Constraints: -
    # @(Inputs): S1 - current storage in S1 [mm]
    #            S1max - maximum storage in S1 [mm]
    #            S2 - current storage in S2 [mm]
    #            S2max - maximum storage in S2 [mm]

    out = ((S2 * S1max - S1 * S2max) / (S1max + S2max)) * smooth_threshold_storage_logistic(S1 / S1max, S2 / S2max)
    return out
