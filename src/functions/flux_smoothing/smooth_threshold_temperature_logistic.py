import numpy as np

def smooth_threshold_temperature_logistic(T, Tt, r=0.01):
    """
    Logistic smoother for temperature threshold functions.

    Smooths the transition of threshold functions of the form:
    Snowfall = { P, if T <  Tt
                { 0, if T >= Tt

    By transforming the equation above to Sf = f(P, T, Tt, r):
    Sf = P * 1 / (1 + exp((T - Tt) / r))

    Inputs:

    T       : current temperature
    Tt      : threshold temperature below which snowfall occurs
    r       : smoothing parameter rho, default = 0.01

    NOTE: this function only outputs the multiplier. This needs to be
    applied to the proper flux (P in Sf equation) outside of this function.
    """
    # Calculate multiplier 
    #bug fix: 11July2024 - SAS - if else statement added to prevent overflow in np.exp term
    
    if (T - Tt) / r >= 700:
        out = 0
    else:
        out = 1 / (1 + np.exp((T - Tt) / r))

    return out