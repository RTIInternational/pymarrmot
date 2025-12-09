import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation.evap_1 import evap_1
from pymarrmot.models.flux.snowfall.snowfall_1 import snowfall_1
from pymarrmot.models.flux.rainfall.rainfall_1 import rainfall_1
from pymarrmot.models.flux.melt.melt_1 import melt_1
from pymarrmot.models.flux.saturation.saturation_1 import saturation_1
from pymarrmot.models.flux.interflow.interflow_8 import interflow_8
from pymarrmot.models.flux.baseflow.baseflow_1 import baseflow_1

class m_12_alpine2_6p_2s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Alpine model v2

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model reference:
    Eder, G., Sivapalan, M., & Nachtnebel, H. P. (2003). Modelling water 
    balances in an Alpine catchment through exploitation of emergent 
    properties over changing time scales. Hydrological Processes, 17(11), 
    2125–2149. http://doi.org/10.1002/hyp.1325
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

        self.par_ranges = np.array([[-3, 5],       # tt [celsius]
                                    [0, 20],      # degree-day-factor [mm/degree celsius/d]
                                    [1, 2000],    # Smax [mm]
                                    [0.05, 0.95], # Field capacity as fraction of Smax [-]
                                    [0, 1],       # time delay of interflow [d-1]
                                    [0, 1]])      # time delay of baseflow [d-1]
        
        self.store_names = ["S1", "S2"]  # Names for the stores
        self.flux_names = ["ps", "pr", "qn", "ea", "qse", "qin", "qbf"]  # Names for the fluxes
        
        self.flux_groups = {"Ea": [4],    # Index or indices of fluxes to add to Actual ET
                           "Q": [5, 6, 7]} # Index or indices of fluxes to add to Streamflow

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
        tt = theta[0]    # Threshold temperature for snowfall/snowmelt [celsius]
        ddf = theta[1]   # Degree-day-factor for snowmelt [mm/d/celsius]
        Smax = theta[2]  # Maximum soil moisture storage [mm]
        Cfc = theta[3]   # Field capacity as fraction of Smax [-]
        tcin = theta[4]  # Interflow runoff coefficient [d-1]
        tcbf = theta[5]  # Baseflow runoff coefficient [d-1]
        
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
        flux_ps = snowfall_1(P, T, tt)
        flux_pr = rainfall_1(P, T, tt)
        flux_qn = melt_1(ddf, tt, T, S1, delta_t)
        flux_ea = evap_1(S2, Ep, delta_t)
        flux_qse = saturation_1(P + flux_qn, S2, Smax)
        flux_qin = interflow_8(S2, tcin, Cfc * Smax)
        flux_qbf = baseflow_1(tcbf, S2)

        # stores ODEs
        dS1 = flux_ps - flux_qn
        dS2 = flux_pr + flux_qn - flux_ea - flux_qse - flux_qin - flux_qbf
        
        # outputs
        dS = np.array([dS1, dS2])
        fluxes = np.array([flux_ps, flux_pr, flux_qn, flux_ea,
                           flux_qse, flux_qin, flux_qbf])
        
        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep
        """
        pass
