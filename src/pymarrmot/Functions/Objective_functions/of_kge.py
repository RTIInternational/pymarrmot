import numpy as np
from pymarrmot.functions.objective_functions import check_and_select
from typing import Tuple

def of_kge(obs: np.array, sim: np.array, idx: np.array=None, w=None) -> Tuple[float, np.array, np.array, np.array]:
    """
    Calculates Kling-Gupta Efficiency of simulated streamflow (Gupta et al, 2009). Ignores time steps with negative flow values.

    Parameters
    ----------
    obs : numpy.array
        Time series of observations [nx1].
    sim : numpy.array
        Time series of simulations [nx1].
    idx : numpy.array, optional
        Optional vector of indices to use for calculation, can be logical vector [nx1] or numeric vector [mx1], with m <= n.
    w : numpy.array, optional
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

    # Update default weights if needed
    if w is None:
        w = w_default
    else:
        if not (w.size == 3 and np.ndim(w) == 1):
            raise ValueError('Weights should be a 3x1 or 1x3 vector.')

    # Calculate components
    c = np.array([np.corrcoef(obs, sim)[0, 1], np.std(sim) / np.std(obs), np.mean(sim) / np.mean(obs)])

    # Calculate value
    val = 1 - np.sqrt((w[0] * (c[0] - 1)) ** 2 + (w[1] * (c[1] - 1)) ** 2 + (w[2] * (c[2] - 1)) ** 2)

    return Tuple[val, c, idx, w]