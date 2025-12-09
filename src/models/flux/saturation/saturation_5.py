
def saturation_5(S, p1, p2, In):
    # Flux function
    # Description: Deficit store: exponential saturation excess based on current storage and a threshold parameter
    # Constraints: S >= 0 prevents numerical issues with complex numbers
    # Inputs: p1 - deficit threshold above which no flow occurs [mm]
    #         p2 - exponential scaling parameter [-]
    #         S - current deficit [mm]
    #         In - incoming flux [mm/d]
    
    return (1 - min(1, (max(S, 0) / p1) ** p2)) * In
