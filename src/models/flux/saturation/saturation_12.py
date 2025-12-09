
def saturation_12(p1, p2, In):
    # Flux function
    # Description: Saturation excess flow from a store with different degrees
    # of saturation (min-max linear variant)
    # Constraints: f >= 0
    # Inputs: p1 - maximum contributing fraction area [-]
    #         p2 - minimum contributing fraction area [-]
    #         In - incoming flux [mm/d]
    
    out = max(0, (p1 - p2) / (1 - p2)) * In

    return out
