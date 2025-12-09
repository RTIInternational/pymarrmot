import numpy as np
from typing import Tuple
from pymarrmot.functions.objective_functions.check_and_select import check_and_select

def of_rmse(obs: np.array, sim: np.array, idx: np.array = None) -> Tuple[float, np.array]:
    """
    Calculates the Root Mean Squared Error of simulated streamflow.
    Ignores time steps with negative flow values.

    Parameters
    ----------
    obs : np.array
        Time series of observations [nx1].
    sim : np.array
        Time series of simulations [nx1].
    idx : np.array, optional
        Optional vector of indices to use for calculation, can be logical vector [nx1] or numeric vector [mx1], with m <= n.

    Returns
    -------
    tuple
        Tuple containing:
        - val : float
            Objective function value [1x1].
        - idx : np.array, optional
            Vector of indices used for the calculation.

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
    val = np.sqrt(np.mean(np.square(obs - sim)))

    return val, idx