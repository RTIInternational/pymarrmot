import numpy as np

def smooth_threshold_storage_logistic(s, smax, r=0.01, e=5.0):
    """
    Logisitic smoother for storage threshold functions.

    Converted from MatLab to Python, 2024, RTI International
    Copyright (C) 2018 Wouter J.M. Knoben
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Smooths the transition of threshold functions of the form:
    Q = { P, if S = Smax
        { 0, if S < Smax

    By transforming the equation above to Q = f(P,S,Smax,e,r):
    Q = P * 1/ (1+exp((S-Smax+r*e*Smax)/(r*Smax)))

    Inputs:
    S       : current storage
    Smax    : maximum storage
    r       : smoothing parameter rho, default = 0.01
    e       : smoothing parameter e, default 5

    NOTE: this function only outputs the multiplier. This needs to be
    applied to the proper flux outside of this function.

    NOTE: can be applied for temperature thresholds as well (i.e. snow
    modules). This simply means that S becomes T, and Smax T0.
    """

    smax = max(0, smax)

    # Calculate multiplier
    #bug fix: 11July2024 - SAS - nested if else statement added to prevent overflow in np.exp term
    if r * smax == 0:
        if (s - smax + r * e * smax) / r >= 700:
            out = 0
        else:
            out = 1 / (1 + np.exp((s - smax + r * e * smax) / r))
    else:
        if (s - smax + r * e * smax) / (r * smax) >= 700:
            out = 0
        else:
            out = 1 / (1 + np.exp((s - smax + r * e * smax) / (r * smax)))

    return out