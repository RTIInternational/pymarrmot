
def exchange_3(p1, S, p2):
    '''
    Flux function
    Description: Water exchange with infinite size store based on threshold
    Constraints: -
    Inputs:
        p1 - base leakage time delay [d-1]
        p2 - threshold for flow reversal [mm]
        S  - current storage [mm]
    '''
    return p1 * (S - p2)
