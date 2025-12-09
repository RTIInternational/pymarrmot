import numpy as np
from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation.evap_6 import evap_6
from pymarrmot.models.flux.evaporation.evap_5 import evap_5
from pymarrmot.models.flux.saturation.saturation_1 import saturation_1
from pymarrmot.models.flux.interflow.interflow_9 import interflow_9
from pymarrmot.models.flux.baseflow.baseflow_1 import baseflow_1

class m_04_newzealand1_6p_1s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: New Zealand model v1

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model reference:
    Atkinson, S. E., Woods, R. A., & Sivapalan, M. (2002). Climate and
    landscape controls on water balance model complexity over changing
    timescales. Water Resources Research, 38(12), 17–50.
    http://doi.org/10.1029/2002WR001487
    """

    def __init__(self):
        super().__init__()
        self.num_stores = 1  # number of model stores
        self.num_fluxes = 5  # number of model fluxes
        self.num_params = 6

        self.jacob_pattern = [1]  # Jacobian matrix of model store ODEs

        self.par_ranges = [
            [1, 2000],      # Smax, Maximum soil moisture storage [mm]
            [0.05, 0.95],   # sfc, Field capacity as fraction of maximum soil moisture [-]
            [0.05, 0.95],   # m, Fraction forest [-]
            [0, 1],         # a, Subsurface runoff coefficient [d-1]
            [1, 5],         # b, Runoff non-linearity [-]
            [0, 1]          # tcbf, Baseflow runoff coefficient [d-1]
        ]

        self.store_names = ["S1"]  # Names for the stores
        self.flux_names = ["veg", "ebs", "qse", "qss", "qbf"]  # Names for the fluxes

        self.flux_groups = {
            "Ea": [1, 2],   # Index or indices of fluxes to add to Actual ET
            "Q": [3, 4, 5]  # Index or indices of fluxes to add to Streamflow
        }

    def init(self):
        pass

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation

        Parameters:
        - S : array_like
            State variables

        Returns:
        - dS : array_like
            State derivatives
        - fluxes : array_like
            Fluxes
        """
        # parameters
        theta = self.theta
        s1max = theta[0]    # Maximum soil moisture storage [mm]
        sfc = theta[1]      # Field capacity as fraction of maximum soil moisture [-]
        m = theta[2]        # Fraction forest [-]
        a = theta[3]        # Subsurface runoff coefficient [d-1]
        b = theta[4]        # Runoff non-linearity [-]
        tcbf = theta[5]     # Baseflow runoff coefficient [d-1]

        # delta_t
        delta_t = self.delta_t

        # stores
        S1 = S[0]

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        # fluxes functions
        flux_veg = evap_6(m, sfc, S1, s1max, Ep, delta_t)
        flux_ebs = evap_5(m, S1, s1max, Ep, delta_t)
        flux_qse = saturation_1(P, S1, s1max)
        flux_qss = interflow_9(S1, a, sfc * s1max, b, delta_t)
        flux_qbf = baseflow_1(tcbf, S1)

        # stores ODEs
        dS1 = P - flux_veg - flux_ebs - flux_qse - flux_qss - flux_qbf

        # outputs
        dS = np.array([dS1])
        fluxes = [flux_veg, flux_ebs, flux_qse, flux_qss, flux_qbf]

        return dS, fluxes

    def step(self):
        pass
