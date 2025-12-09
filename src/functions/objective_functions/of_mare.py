import numpy as np
from pymarrmot.functions.objective_functions.check_and_select import check_and_select
from typing import Tuple

def of_mare(obs: np.array, sim: np.array, idx: np.array=None) -> Tuple[float, np.array]:
    """
    Calculates the Mean Absolute Relative Error of simulated streamflow.
    Ignores time steps with negative flow values. Adds a constant e of 1/100 of mean(obs) to avoid issues with zero flows (Pushpalatha et al., 2012).

    Parameters
    ----------
    obs : numpy.array
        Time series of observations [nx1].
    sim : numpy.array
        Time series of simulations [nx1].
    idx : numpy.array, optional
        Optional vector of indices to use for calculation, can be logical vector [nx1] or numeric vector [mx1], with m <= n.

    Returns
    -------
    tuple
        Tuple containing:
        - val : float
            Objective function value [1x1].
        - idx : numpy.array, optional
            Vector of indices used for calculation.

    References
    ----------
    Pushpalatha, R.; Perrin, C.; le Moine, N. and Andréassian V. (2012). "A
    review of efficiency criteria suitable for evaluating low-flow
    simulations". Journal of Hydrology. 420-421, 171-182.
    doi:10.1016/j.jhydrol.2011.11.055
    """

    # Check inputs and select timesteps
    if obs.size < 2 or sim.size < 2:
        raise ValueError('Not enough input arguments')

    if idx is None:
        idx = np.array([])

    # Check and select data
    sim, obs, idx = check_and_select(sim, obs, idx)

    # Find the constant e
    e = np.mean(obs) / 100

    # Apply constant and transform flows
    obs = obs + e
    sim = sim + e

    # Calculate metric
    val = np.mean(np.abs((sim - obs) / obs))

    return val, idx