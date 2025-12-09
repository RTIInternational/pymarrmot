
# Flux function
# ------------------
# Description:  Exponential inflow to surface depression store
# Constraints:  f <= (Smax-S)/dt
#               S <= Smax
# @(Inputs):    p1   - linear scaling parameter [-]
#               p2   - exponential scaling parameter [-]
#               S    - current storage [mm]
#               Smax - maximum storage [mm]
#               dt   - time step size [d]

import numpy as np

def depression_1(p1, p2, S, Smax, flux, dt):
    out = min(p1 * np.exp(-1 * p2 * S / max(Smax - S, 0)) * flux, max((Smax - S) / dt, 0))
    return out
