import numpy as np
from typing import Tuple
from pymarrmot.functions.objective_functions.check_and_select import check_and_select

def of_root5_kge(obs: np.array, sim: np.array, idx: np.array = None, w: np.array = None) -> Tuple[float, np.array, np.array, np.array]:
    """
    Calculates Kling-Gupta Efficiency of the fith root of simulated streamflow,
    intended to capture low flow aspects better. Ignores time steps with obs < 0 values.

    Parameters
    ----------
    obs : np.array
        Time series of observations [nx1].
    sim : np.array
        Time series of simulations [nx1].
    idx : np.array, optional
        Optional vector of indices to use for calculation, can be logical vector [nx1] or numeric vector [mx1], with m <= n.
    w : np.array, optional
        Optional weights of components [3x1].

    Returns
    -------
    tuple
        Tuple containing:
        - val : float
            Objective function value [1x1].
        - c : np.array
            Components [r, alpha, beta] [3x1].
        - idx : np.array, optional
            Indices used for the calculation.
        - w : np.array
            Weights [wr, wa, wb] [3x1].

    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE
    performance criteria: Implications for improving hydrological modelling. Journal of Hydrology, 377(1–2), 80–91.
    https://doi.org/10.1016/j.jhydrol.2009.08.003

    Chiew, F.H.S., Stewardson, M.J., McMahon, T.A., 1993. Comparison of six rainfall-runoff modelling approaches. J. Hydrol.
    https://doi.org/10.1016/0022-1694(93)90073-I
    """

    # Check inputs and select timesteps
    if obs.size < 2 or sim.size < 2:
        raise ValueError('Not enough input arguments')

    if idx is None:
        idx = np.array([])

    # Check and select data
    sim, obs, idx = check_and_select(sim, obs, idx)

    # Set weights
    w_default = np.array([1, 1, 1])  # default weights
    if w is None:
        w = w_default
    else:
        if w.shape != (3,) and w.shape != (1, 3):
            raise ValueError('Weights should be a 3x1 or 1x3 vector.')

    # Calculate the fifth root of the flows
    sim[sim < 0] = 0
    obs = np.power(obs, 0.2)
    sim = np.power(sim, 0.2)

    # Calculate components
    c = np.zeros(3)
    c[0] = np.corrcoef(obs, sim)[0, 1]  # r: linear correlation
    c[1] = np.std(sim) / np.std(obs)  # alpha: ratio of standard deviations
    c[2] = np.mean(sim) / np.mean(obs)  # beta: bias

    # Calculate value
    val = 1 - np.sqrt((w[0] * (c[0] - 1))**2 + (w[1] * (c[1] - 1))**2 + (w[2] * (c[2] - 1))**2)  # weighted KGE

    return val, c, idx, w