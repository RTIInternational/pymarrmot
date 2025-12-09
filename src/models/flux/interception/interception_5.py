
def interception_5(p1, p2, In):
    # Flux function
    # ------------------
    # Description: Interception excess after a combined absolute amount and fraction are intercepted
    # Constraints: f >= 0
    # @(Inputs): p1 - fraction that is not throughfall [-]
    #            p2 - constant interception and evaporation [mm/d]
    #            In - incoming flux [mm/d]
    
    out = max(p1 * In - p2, 0)
    
    return out
