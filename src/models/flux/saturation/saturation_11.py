from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic

def saturation_11(p1, p2, S, Smin, Smax, In, *args):
    '''
    saturation_11

    Description:
    Saturation excess flow from a store with different degrees of saturation (min exponential variant)

    Constraints:
    f <= In

    Inputs:
    p1   - linear scaling parameter [-]
    p2   - exponential scaling parameter [-]
    S    - current storage [mm]
    Smin - minimum contributing storage [mm]
    Smax - maximum contributing storage [mm]
    In   - incoming flux [mm/d]
    args:
        smoothing variable r (default 0.01)
        smoothing variable e (default 5.00)
    '''

    if len(args) == 0:
        return In * min(1, p1 * (max(0, S - Smin) / (Smax - Smin)) ** p2) * (1 - smooth_threshold_storage_logistic(S, Smin))
    elif len(args) == 1:
        return In * min(1, p1 * (max(0, S - Smin) / (Smax - Smin)) ** p2) * (1 - smooth_threshold_storage_logistic(S, Smin, args[0]))
    elif len(args) == 2:
        return In * min(1, p1 * (max(0, S - Smin) / (Smax - Smin)) ** p2) * (1 - smooth_threshold_storage_logistic(S, Smin, args[0], args[1]))
