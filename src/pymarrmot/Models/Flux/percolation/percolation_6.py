import numpy as np

def percolation_6(p1, p2, S, dt):
    """
    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).

    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Flux function

    Description:  Threshold-based percolation from a store that can reach negative values
    Constraints:  f <= S/dt

    Parameters
    ----------
    p1 : float
        maximum percolation rate
    p2 : float
        storage threshold for reduced percolation [mm]
    S : float
        current storage [mm]
    dt : float
        time step size [d]

    Returns
    -------
    out : float
        Percolation from storage
    """
    out = min(max(0,S)/dt, p1*min(1,max(0,S)/p2))
    return out