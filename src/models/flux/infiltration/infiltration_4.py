
def infiltration_4(fluxIn, p1):
    # Flux function
    # Description:  Constant infiltration rate 
    # Constraints:  f <= fin
    # Inputs:       p1   - Infiltration rate [mm/d]
    #               fin  - incoming flux [mm/d]
    
    # Returns the minimum of fluxIn and p1
    return min(fluxIn, p1)
