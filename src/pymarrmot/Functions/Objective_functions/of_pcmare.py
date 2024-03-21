import numpy as np
from typing import Tuple
import check_and_select

def of_pcmare(obs: np.ndarray, sim: np.ndarray, idx: np.ndarray = None) -> Tuple[float, np.ndarray]:
    """
    Calculates a version of the Mean Absolute Relative Error (MARE) of simulated streamflow as a percentage of the MARE of the mean observed flow.
    Ignores time steps with negative flow values. Adds a constant e of 1/100 of mean(obs) to avoid issues with zero flows.

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
    Pushpalatha, R.; Perrin, C.; le Moine, N. and Andréassian V. (2012). "A review of efficiency criteria suitable for evaluating low-flow
    simulations". Journal of Hydrology. 420-421, 171-182. doi:10.1016/j.jhydrol.2011.11.055.

    This implementation is based on the original MATLAB code.
    """

    # Check inputs and select timesteps
    if obs.size < 2 or sim.size < 2:
        raise ValueError('Not enough input arguments')

    if idx is None:
        idx = np.array([])

    # Check and select data
    sim, obs, idx = check_and_select(sim, obs, idx)

    # Find the constant e
    m = np.mean(obs)
    e = m / 100

    # Apply constant and transform flows
    obs = obs + e
    sim = sim + e

    # Calculate metric
    mare = np.mean(np.abs((sim - obs) / obs))
    mare_mean = np.mean(np.abs((sim - m) / m))

    val = mare / mare_mean

    return val, idx
