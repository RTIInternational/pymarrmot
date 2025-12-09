import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation.evap_6 import evap_6
from pymarrmot.models.flux.evaporation.evap_5 import evap_5
from pymarrmot.models.flux.saturation.saturation_1 import saturation_1
from pymarrmot.models.flux.interflow.interflow_7 import interflow_7
from pymarrmot.models.flux.baseflow.baseflow_1 import baseflow_1
from pymarrmot.models.flux.baseflow.baseflow_2 import baseflow_2

class m_09_susannah1_6p_2s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Susannah Brook v1

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model reference:
    Son, K., & Sivapalan, M. (2007). Improving model structure and reducing 
    parameter uncertainty in conceptual water balance models through the use 
    of auxiliary data. Water Resources Research, 43(1). 
    https://doi.org/10.1029/2006WR005032
    """

    def __init__(self):
        """
        creator method
        """
        super().__init__()
        self.num_stores = 2  # number of model stores
        self.num_fluxes = 7  # number of model fluxes
        self.num_params = 6

        self.jacob_pattern = np.array([[1, 0],
                                       [1, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[1, 2000],    # Sb, Maximum soil moisture storage [mm]
                                    [0.05, 0.95],  # Sfc, Wilting point as fraction of sb [-]
                                    [0.05, 0.95],  # M, Fraction forest [-]
                                    [1, 50],       # a, Runoff coefficient [d] (should be > 0)
                                    [0.2, 1],      # b, Runoff coefficient [-] (should be > 0)
                                    [0, 1]])       # r, Runoff coefficient [d-1]

        self.store_names = ["S1", "S2"]  # Names for the stores
        self.flux_names = ["Ebs", "Evg", "Qse", "Qss",
                          "qr", "Qb", "qt"]  # Names for the fluxes

        self.flux_groups = {"Ea": [1, 2],  # Index or indices of fluxes to add to Actual ET
                           "Q": 7}        # Index or indices of fluxes to add to Streamflow

    def init(self):
        """
        INITialisation function
        """
        pass

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation

        Parameters:
            S (numpy.array): State variables

        Returns:
            tuple: Tuple containing the change in state variables and fluxes
        """
        # parameters
        theta = self.theta
        sb = theta[0]   # Maximum soil moisture storage [mm]
        sfc = theta[1]  # Wiliting point as fraction of sb [-]
        m = theta[2]    # Fraction forest [-]
        a = theta[3]    # Runoff coefficient [d]
        b = theta[4]    # Runoff coefficient [-]
        r = theta[5]    # Runoff coefficient [d-1]

        # delta_t
        delta_t = self.delta_t

        # stores
        S1 = S[0]
        S2 = S[1]

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        # fluxes functions
        flux_ebs = evap_5(m, S1, sb, Ep, delta_t)
        flux_eveg = evap_6(m, sfc, S1, sb, Ep, delta_t)
        flux_qse = saturation_1(P, S1, sb)
        flux_qss = interflow_7(S1, sb, sfc, a, b, delta_t)
        flux_qr = baseflow_1(r, flux_qss)
        flux_qb = baseflow_2(S2, a, b, delta_t)
        flux_qt = flux_qse + (flux_qss - flux_qr) + flux_qb

        # stores ODEs
        dS1 = P - flux_ebs - flux_eveg - flux_qse - flux_qss
        dS2 = flux_qr - flux_qb

        # outputs
        dS = np.array([dS1, dS2])
        fluxes = np.array([flux_ebs, flux_eveg, flux_qse, flux_qss,
                           flux_qr, flux_qb, flux_qt])

        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep
        """
        pass
