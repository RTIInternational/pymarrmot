# Equivalent Python function for the Matlab evaporation function

# Threshold-based evaporation with constant minimum rate
# Arguments:
# p1 : float
#     Wilting point (1st threshold) [mm]
# p2 : float
#     2nd threshold as a fraction of wilting point [-]
# S : float
#     Current storage [mm]
# Ep : float
#     Potential evapotranspiration rate [mm/d]
# dt : float
#     Time step size [d]
# Returns:
# float
#     Evaporation rate
# Function Body:
def evap_21(p1, p2, S, Ep, dt):
    return min(max(p2, min(S / p1, 1)) * Ep, S / dt)
