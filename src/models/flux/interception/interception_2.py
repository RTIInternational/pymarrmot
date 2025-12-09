
def interception_2(In, p1):
    # Flux function
    # Description: Interception excess after a constant amount is intercepted
    # Constraints: - f >= 0
    # Inputs: In - incoming flux [mm/d]
    #            p1   - interception and evaporation capacity [mm/d]
    # Outputs: f - interception excess [mm/d]

    out = max(In - p1, 0)
    return out
