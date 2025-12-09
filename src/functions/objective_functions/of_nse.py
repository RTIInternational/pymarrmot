import numpy as np
from pymarrmot.functions.objective_functions.check_and_select import check_and_select
from typing import Tuple

def of_nse(obs: np.ndarray, sim: np.ndarray, idx: np.ndarray = None) -> Tuple[float, np.ndarray]:
    """
    Calculates the Nash-Sutcliffe Efficiency of simulated streamflow (Nash & Sutcliffe, 1970).
    Ignores time steps with negative flow values.

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
            Indices used for the calculation.

    References
    ----------
    Nash, J. E.; Sutcliffe, J. V. (1970). "River flow forecasting through 
    conceptual models part I — A discussion of principles". Journal of 
    Hydrology. 10 (3): 282–290. Bibcode:1970JHyd...10..282N. 
    doi:10.1016/0022-1694(70)90255-6.

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
    top = np.sum((sim - obs)**2)
    bot = np.sum((obs - np.mean(obs))**2)
    val = 1 - (top / bot)

    return val, idx