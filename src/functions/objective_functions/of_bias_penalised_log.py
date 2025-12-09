import numpy as np
from pymarrmot.functions.objective_functions.check_and_select import check_and_select
from typing import Tuple

def of_bias_penalised_log(obs: np.array, sim: np.array, idx: np.array, of_name: str, *varargs: tuple) -> Tuple[float, np.array]:
    """
    OF_BIAS_PENALISED_LOG applies a bias penalisation to an objective function as suggested by Viney et al. (2009).
    Ignore timesteps with negative values.

    Given the efficiency of the OF specified (E_of):
        E = E_of - 5 * abs(np.log(1 + B)) ** 2.5
    where B = mean(sim - obs) / mean(obs) is the bias between obs and sim

    Parameters
    ----------
    obs : list
        Time series of observations [nx1].
    sim : list
        Time series of simulations [nx1].
    idx : list
        Optional vector of indices to use for calculation, can be logical vector [nx1] or numeric vector [mx1],
        with m <= n.
    of_name : str
        String name of the objective function to penalise.
    *varargs : tuple
        Additional arguments to of_name.

    Returns
    -------
    val : float
        Objective function value [1x1].
    idx : list
        Indices used for the calculation.
    """
    # Check inputs
    if len(of_name) == 0:
        raise ValueError('Not enough input arguments')

    if idx is None:
        idx = []

    # Check and select data
    sim, obs, idx = check_and_select(sim, obs, idx)

    # Calculate components
    e_of = of_name(obs, sim, *varargs)
    B = np.mean(sim - obs) / np.mean(obs)

    # Calculate value
    val = e_of - 5 * abs(np.log(1 + B)) ** 2.5

    return val, idx