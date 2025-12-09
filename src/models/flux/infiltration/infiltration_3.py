from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic

def infiltration_3(In, S, Smax, *varargs):
    '''
    Infiltration to soil moisture of liquid water stored in snow pack

    Arguments:
    In : float
        incoming flux [mm/d]
    S : float
        current storage [mm]
    Smax : float
        maximum storage [mm]
    varargs : tuple
        smoothing variable r (default 0.01), smoothing variable e (default 5.00)

    Returns:
    out : float
    '''
    if len(varargs) == 0:
        out = In * (1 - smooth_threshold_storage_logistic(S, Smax))
    elif len(varargs) == 1:
        out = In * (1 - smooth_threshold_storage_logistic(S, Smax, varargs[0]))
    elif len(varargs) == 2:
        out = In * (1 - smooth_threshold_storage_logistic(S, Smax, varargs[0], varargs[1]))
    return out
