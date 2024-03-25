import numpy as np
from pymarrmot.functions.objective_functions import check_and_select
from pymarrmot.functions.objective_functions import of_kge
from pymarrmot.functions.objective_functions import of_root5_kge
from typing import Tuple

def of_mean_hilo_root5_KGE(obs: np.array, sim: np.array, idx: np.array=None, w: np.array=None) -> Tuple[float, np.array, np.array, np.array]:
    """
    Calculates the average Kling-Gupta Efficiency of the untransformed and fifth root of streamflow
    (Gupta et al., 2009) of the untransformed and fifth root (Chiew et al., 1993) of
    streamflow. Ignores time steps with negative flow values.

    Parameters
    ----------
    obs : numpy.array
        Time series of observations [nx1].
    sim : numpy.array
        Time series of simulations [nx1].
    idx : numpy.array, optional
        Optional vector of indices to use for calculation, can be logical vector [nx1] or numeric vector [mx1], with m <= n.
    w : numpy.array, optional
        Optional weights of components. If one set of weights is provided, it is used for both untransformed and fifth root KGE.
        If a list of two sets of weights is provided, they are used for high and low KGE, respectively.

    Returns
    -------
    tuple
        Tuple containing:
        - val : float
            Objective function value [1x1].
        - c : numpy.array
            Components [r, alpha, beta] from high and low KGE.
        - idx : numpy.array, optional
            Indices used for the calculation.
        - w : numpy.array
            Weights [wr, wa, wb] from high and low KGE.

    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009).
    Decomposition of the mean squared error and NSE performance criteria:
    Implications for improving hydrological modelling. Journal of Hydrology,
    377(1–2), 80–91. https://doi.org/10.1016/j.jhydrol.2009.08.003

    Chiew, F.H.S., Stewardson, M.J., McMahon, T.A., 1993. Comparison of six
    rainfall-runoff modelling approaches. J. Hydrol.
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
    if w is None:
        w1 = np.array([1, 1, 1])
        w2 = np.array([1, 1, 1])
    else:
        if isinstance(w, list) and len(w) == 2:
            w1, w2 = w[0], w[1]
        elif isinstance(w, np.ndarray) and w.ndim == 1 and w.size == 3:
            w1, w2 = w, w
        else:
            raise ValueError('w should be either a list of size 2 or a numpy array of size [1,3]')

    # Call individual KGE functions
    val1, c1, _, w1 = of_kge(obs, sim, None, w1)
    val2, c2, _, w2 = of_root5_kge(obs, sim, None, w2)

    # Calculate value
    val = 0.5 * (val1 + val2)  # Weighted KGE
    c = [c1, c2]  # Components from high and low KGE
    w = [w1, w2]  # Weights from high and low KGE

    return val, c, idx, w