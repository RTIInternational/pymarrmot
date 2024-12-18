import numpy as np
from scipy.integrate import quad

def uh_5_half(d_base, delta_t):
    """
    Unit Hydrograph [days] with half a triangle (exponential decay)

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Parameters:
    - d_base : float
        Time base of routing delay [d]
    - delta_t : float
        Time step size [d]

    Returns:
    - UH : array_like
        Unit hydrograph [nx2], where UH's first row contains coefficients to split
        flow at each of n timesteps forward, and the second row contains zeros now,
        which are the still-to-flow values.

    Unit hydrograph spreads the input volume over a time period delay.
    Percentage of input returned only decreases.
    I.e. d_base = 3.8 [days], delta_t = 1:
    UH(1) = 0.841  [% of inflow]
    UH(2) = 0.133
    UH(3) = 0.021
    UH(4) = 0.004
    """
    # TIME STEP SIZE
    delay = d_base / delta_t
    if delay == 0:
        delay = 1  # any value below t = 1 means no delay, but zero leads to problems
    tt = np.arange(1, np.ceil(delay) + 1)  # time series for which we need UH ordinates [days]

    # UNIT HYDROGRAPH
    # The Unit Hydrograph follows exponential decay y=exp(-x). For a given delay time,
    # the fraction of flow per time step is thus the integral of t-1 to t of the exponential decay curve.
    # The curve has range [0, Inf>. We impose the arbitrary boundary of [0, 7] here (exp(-7) = 9e-4) as the
    # point where the decay curve 'ends'. This allows to divide the range [0, 7] in n delay steps,
    # and so calculate the UH.

    # Find integral limits
    stepsize = (7 - 0) / delay  # Range over which the decay curve is calculated, divided by required number of delay steps
    limits = np.arange(0, 7 + stepsize, stepsize)

    # UNIT HYDROGRAPH
    UH = np.zeros(len(tt))

    for t in range(len(tt)):
        UH[t] = quad(lambda x: np.exp(-x), limits[t], limits[t + 1])[0]

    # ACCOUNT FOR <7,Inf> PART OF THE CURVE (i.e. add the missing tail end of the curve to the last delay step,
    # to ensure that 100% of flow is routed).
    UH[-1] += 1 - np.sum(UH)
    zeros = np.zeros_like(UH)
    UH = np.vstack([UH, zeros])
    return UH