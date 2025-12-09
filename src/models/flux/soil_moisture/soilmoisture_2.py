from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic

def soilmoisture_2(S1, S1max, S2, S2max, S3, S3max):
    # Description: Water rebalance to equal relative storage (3 stores)
    # Constraints: -
    # @(Inputs):   S1    - current storage in S1 [mm]
    #             S1max - maximum storage in S1 [mm]
    #             S2    - current storage in S2 [mm]
    #             S2max - maximum storage in S2 [mm]
    #             S3    - current storage in S3 [mm]
    #             S3max - maximum storage in S3 [mm]
    
    out = (S2 * (S1 * (S2max + S3max) + S1max * (S2 + S3)) / ((S2max + S3max) * (S1max + S2max + S3max))) * smooth_threshold_storage_logistic(S1 / S1max, (S2 + S3) / (S2max + S3max))
    
    return out
