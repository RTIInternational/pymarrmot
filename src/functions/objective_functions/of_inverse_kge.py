import numpy as np
from pymarrmot.functions.objective_functions.check_and_select import check_and_select
from typing import Tuple

def of_inverse_kge(obs: np.array, sim: np.array, idx: np.array=None, w: float=None):
    """
    OF_INVERSE_KGE Calculates Kling-Gupta Efficiency of the inverse of simulated streamflow (Gupta et al, 2009),
    intended to capture low flow aspects better (Pushpalatha et al, 2012). Ignores time steps with -999 values.

    Parameters
    ----------
    obs : array
        Time series of observations [nx1].
    sim : array
        Time series of simulations [nx1].
    idx : array, optional
        Optional vector of indices to use for calculation, can be logical vector [nx1] or numeric vector [mx1], with m <= n.
    w : array, optional
        Optional weights of components [3x1].

    Returns
    -------
    tuple
        Tuple containing:
        - val : float
            Objective function value [1x1].
        - c : list
            Components [r, alpha, beta] from high and low KGE.
        - idx : numpy.array, optional
            Indices used for the calculation.
        - w : list
            Weights [wr, wa, wb] from high and low KGE.
    
    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009).
    Decomposition of the mean squared error and NSE performance criteria:
    Implications for improving hydrological modelling. Journal of Hydrology,
    377(1–2), 80–91. https://doi.org/10.1016/j.jhydrol.2009.08.003

    Pushpalatha, R., Perrin, C., Moine, N. Le, & Andréassian, V. (2012). A
    review of efficiency criteria suitable for evaluating low-flow
    simulations. Journal of Hydrology, 420–421, 171–182.
    https://doi.org/10.1016/j.jhydrol.2011.11.055
    """

    # Check inputs and select timesteps
    if len(obs) < 2 or len(sim) < 2:
        raise ValueError('Not enough input arguments')

    if idx is None:
        idx = []

    # Check and select data
    sim, obs, idx = check_and_select(sim, obs, idx)

    # Set weights
    w_default = [1, 1, 1]  # default weights

    # Update default weights if needed
    if w is None:
        w = w_default
    else:
        if not (len(w) == 3):
            raise ValueError('Weights should be an array of length 3.')

    # Invert the time series and add a small constant to avoid issues with 0 flows
    e = np.mean(obs) / 100
    obs = 1 / (obs + e)
    sim = 1 / (sim + e)

    # Calculate components
    c = [np.corrcoef(obs, sim)[0, 1], np.std(sim) / np.std(obs), np.mean(sim) / np.mean(obs)]

    # Calculate value
    val = 1 - np.sqrt((w[0] * (c[0] - 1)) ** 2 + (w[1] * (c[1] - 1)) ** 2 + (w[2] * (c[2] - 1)) ** 2)

    return val, c, idx, w