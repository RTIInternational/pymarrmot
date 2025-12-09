
def exchange_2(p1, S1, S1max, S2, S2max):
    # Water exchange based on relative storages
    # Inputs: p1 - base exchange rate [mm/d]
    #         S1 - current storage in S1 [mm]
    #         S1max - maximum storage in S1 [mm]
    #         S2 - current storage in S2 [mm]
    #         S2max - maximum storage in S2 [mm]
    
    out = p1 * (S1/S1max - S2/S2max)
    return out
