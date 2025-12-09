import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.baseflow import baseflow_2
from pymarrmot.models.flux.evaporation import evap_3, evap_7
from pymarrmot.models.flux.split import split_1
from pymarrmot.models.flux.saturation import saturation_1
from pymarrmot.models.flux.interflow import interflow_9
from pymarrmot.models.unit_hydro import (route, uh_4_full, update_uh)

class m_11_collie3_6p_2s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Collie river v3

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model reference:
    Jothityangkoon, C., M. Sivapalan, and D. Farmer (2001), Process controls
    of water balance variability in a large semi-arid catchment: downward 
    approach to hydrological model development. Journal of Hydrology, 254,
    174-198. doi: 10.1016/S0022-1694(01)00496-6.
    """

    def __init__(self, delta_t=None, theta=None):
        """
        creator method
        """
        super().__init__()
        self.num_stores = 2  # number of model stores
        self.num_fluxes = 7  # number of model fluxes
        self.num_params = 6
        
        self.jacob_pattern = np.array([[1, 0],
                                      [1, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[1, 2000],   # Smax [mm]
                                    [0.05, 0.95],  # fc as fraction of Smax [-] 
                                    [0, 1],       # a, subsurface runoff coefficient [d-1]
                                    [0.05, 0.95],  # M, fraction forest cover [-]
                                    [1, 5],       # b, flow non-linearity [-]
                                    [0, 1]])      # lambda, flow distribution [-]
        
        self.store_names = ["S1", "S2"]  # Names for the stores
        self.flux_names = ["eb", "ev", "qse", "qss", "qsss", "qsg", "qt"]  # Names for the fluxes
        
        self.flux_groups = {"Ea": [1, 2],   # Index or indices of fluxes to add to Actual ET
                           "Q": 7}         # Index or indices of fluxes to add to Streamflow

        # setting delta_t and theta triggers the function obj.init()
        if delta_t is not None:
            self.delta_t = delta_t
        if theta is not None:
            self.theta = theta

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
        S1max = theta[0]    # Maximum soil moisture storage [mm] 
        Sfc = theta[1]      # Field capacity as fraction of S1max [-] 
        a = theta[2]        # Subsurface runoff coefficient [d-1]
        M = theta[3]        # Fraction forest cover [-]
        b = theta[4]        # Non-linearity coefficient [-]
        lambda_ = theta[5]  # Flow distribution parameter [-]
        
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
        flux_eb = evap_7(S1, S1max, (1 - M) * Ep, delta_t)
        flux_ev = evap_3(Sfc, S1, S1max, M * Ep, delta_t)
        flux_qse = saturation_1(P, S1, S1max)
        flux_qss = interflow_9(S1, a, Sfc * S1max, b, delta_t)
        flux_qsss = split_1(lambda_, flux_qss)
        flux_qsg = baseflow_2(S2, 1 / a, 1 / b, delta_t)
        flux_qt = flux_qse + (1 - lambda_) * flux_qss + flux_qsg
        
        # stores ODEs
        dS1 = P - flux_eb - flux_ev - flux_qse - flux_qss
        dS2 = flux_qsss - flux_qsg
         
        # outputs
        dS = np.array([dS1, dS2])
        fluxes = np.array([flux_eb, flux_ev, flux_qse,
                           flux_qss, flux_qsss, flux_qsg, flux_qt])
        
        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep
        """
        pass
