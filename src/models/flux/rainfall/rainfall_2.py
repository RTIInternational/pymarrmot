
def rainfall_2(In, T, p1, p2):
    # Rainfall based on a temperature threshold interval
    # Constraints:  -
    # @(Inputs):    In   - incoming precipitation flux [mm/d]
    #               T    - current temperature [oC]
    #               p1   - midpoint of the combined rain/snow interval [oC]
    #               p2   - length of the mixed snow/rain interval [oC]
    
    out = min(In, max(0, In * (T - (p1 - 0.5 * p2)) / p2))
    return out
