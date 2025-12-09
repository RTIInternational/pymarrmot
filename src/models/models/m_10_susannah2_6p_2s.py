import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation.evap_7 import evap_7
from pymarrmot.models.flux.interflow.interflow_3 import interflow_3
from pymarrmot.models.flux.saturation.saturation_1 import saturation_1
from pymarrmot.models.flux.excess_1 import excess_1

class m_10_susannah2_6p_2s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Susannah Brook model v2

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
        self.num_fluxes = 8  # number of model fluxes
        self.num_params = 6
        
        self.jacob_pattern = np.array([[1, 1],
                                       [1, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[1, 2000],    # Sb, Maximum soil moisture storage [mm]
                                    [0.05, 0.95],  # phi, Porosity [-]
                                    [0.05, 0.95],  # Sfc, Wilting point as fraction of sb [-]
                                    [0, 1],        # r, Fraction of runoff coefficient [-]
                                    [0, 1],        # c, Subsurface flow constant [1/d] (should be > 0)
                                    [1, 5]])       # d, Subsurface flow constant [-] (should be > 0)

        self.store_names = ["S1", "S2"]  # Names for the stores
        self.flux_names = ["eus", "rg", "se", "esat",
                          "qse", "qss", "qr", "qt"]  # Names for the fluxes

        self.flux_groups = {"Ea": [1, 4],   # Index or indices of fluxes to add to Actual ET
                           "Q": 8,         # Index or indices of fluxes to add to Streamflow
                           "GWsink": 7}   # Index or groundwater sink flux

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
        phi = theta[1]  # Porosity [-]
        fc = theta[2]   # Field capacity as fraction of sb [-]
        r = theta[3]    # Fraction of recharge coefficient [-]
        c = theta[4]    # Subsurface flow constant [1/d]
        d = theta[5]    # Subsurface flow constant [-]

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
        flux_eus = evap_7(S1, sb, Ep, delta_t)
        flux_rg = saturation_1(P, S1, (sb - S2) * fc / phi)
        flux_se = excess_1(S1, (sb - S2) * fc / phi, delta_t)
        flux_esat = evap_7(S2, sb, Ep, delta_t)
        flux_qse = saturation_1(flux_rg + flux_se, S2, sb)
        flux_qss = interflow_3((1 - r) * c, d, S2, delta_t)
        flux_qr = interflow_3(r * c, d, S2, delta_t)
        flux_qt = flux_qse + flux_qss

        # stores ODEs
        dS1 = P - flux_eus - flux_rg - flux_se
        dS2 = flux_rg + flux_se - flux_esat - flux_qse - flux_qss - flux_qr

        # outputs
        dS = np.array([dS1, dS2])
        fluxes = np.array([flux_eus, flux_rg, flux_se, flux_esat,
                           flux_qse, flux_qss, flux_qr, flux_qt])

        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep
        """
        pass
