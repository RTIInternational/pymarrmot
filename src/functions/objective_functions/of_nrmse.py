import numpy as np
from typing import Tuple
from pymarrmot.functions.objective_functions.check_and_select import check_and_select

def of_nrmse(obs: np.ndarray, sim: np.ndarray, idx: np.ndarray = None) -> Tuple[float, np.array]:
    """
    Calculates the Normalized Root Mean Squared Error of simulated streamflow,
    with normalization by the range of observations. Ignores time steps with negative flow values.

    Parameters
    ----------
    obs : numpy.ndarray
        Time series of observations [nx1].
    sim : numpy.ndarray
        Time series of simulations [nx1].
    idx : numpy.ndarray, optional
        Optional vector of indices to use for calculation, can be logical vector [nx1] or numeric vector [mx1], with m <= n.

    Returns
    -------
    tuple
        Tuple containing:
        - val : float
            Objective function value [1x1].
        - idx : numpy.ndarray, optional
            Vector of indices used for calculation.

    References
    ----------
    This implementation is based on the original MATLAB code.
    """

    # Check inputs and select timesteps
    if obs.size < 2 or sim.size < 2:
        raise ValueError('Not enough input arguments')

    if idx is None:
        idx = np.array([])

    # Check and select data
    sim, obs, idx = check_and_select(sim, obs, idx)

    # Calculate metric
    obs_range = np.max(obs) - np.min(obs)
    val = np.sqrt(np.mean((obs - sim)**2)) / obs_range

    return val, idx