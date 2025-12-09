import numpy as np
from typing import Tuple

def check_and_select(sim: np.array, obs: np.array, idx: np.array=[None]) -> Tuple[np.array]:
    """
    Checks that sim and obs have the same number of elements, then filters them based on the values in idx AND on the steps
    where obs >= 0.

    Parameters
    ----------
    sim : array
        Time series of simulations.
    obs : array
        Time series of observations.
    idx : array, optional
        Optional array of indices to use for calculation, can be logical vector [nx1] or numeric vector [mx1], with m <= n.

    Returns
    -------
    sim : array
        Time series of simulations [mx1].
    obs : array
        Time series of observations [mx1].
    idx : array
        Vector of indices used to subset [mx1].
    """

    # 1. Check size of inputs:
    # make sure inputs have the same size
    if len(obs) != len(sim):
        raise ValueError('Time series not of equal size.')

    # 2. Get indices where obs >= 0
    idx_exists = [i for i in range(len(obs)) if obs[i] >= 0]

    # 3. Update those if needed with user-input indices
    if idx == [None] or len(idx) == 0:
        idx = idx_exists
    else:
        if isinstance(idx, np.ndarray) and idx.size == len(obs):
            idx = np.intersect1d(np.where(idx)[0], idx_exists)
        elif isinstance(idx, (list, tuple, np.ndarray)):
            idx = np.intersect1d(idx, idx_exists)
        else:
            raise ValueError("Indices should be either "
                             "a logical vector of the same size of Qsim and Qobs, or "
                             "a numeric vector of indices")

    # 4. Filter array based on indices provided in idx
    obs = obs[idx]
    sim = sim[idx]

    return np.array(sim), np.array(obs), np.array(idx)