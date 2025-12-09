
def recharge_5(p1, p2, S1, S2):
    # Flux function
    # Description: Recharge to fulfil evaporation demand if the receiving store is below a threshold
    # Constraints: -
    # @(Inputs): p1 - time coefficient [d-1]
    #            p2 - storage threshold in S2 [mm]
    #            S1 - current storage in S1 [mm]
    #            S2 - current storage in S1 [mm]
    
    out = p1 * S1 * (1 - min(1, S2 / p2))
    
    return out