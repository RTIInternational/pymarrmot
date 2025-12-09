import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation.evap_1 import evap_1
from pymarrmot.models.flux.split.split_1 import split_1
from pymarrmot.models.flux.saturation.saturation_2 import saturation_2
from pymarrmot.models.flux.capillary.capillary_2 import capillary_2
from pymarrmot.models.flux.baseflow.baseflow_1 import baseflow_1
from pymarrmot.models.flux.interception.interception_2 import interception_2
from pymarrmot.models.unit_hydro.route import route
from pymarrmot.models.unit_hydro.uh_3_half import uh_3_half
from pymarrmot.models.unit_hydro.update_uh import update_uh

class m_13_hillslope_7p_2s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Hillslope model

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model reference:
    Savenije, H. H. G. (2010). “Topography driven conceptual modelling 
    (FLEX-Topo).” Hydrology and Earth System Sciences, 14(12), 2681–2692. 
    https://doi.org/10.5194/hess-14-2681-2010
    """

    def __init__(self, delta_t=None, theta=None):
        """
        creator method
        """
        super().__init__()
        self.num_stores = 2  # number of model stores
        self.num_fluxes = 10  # number of model fluxes
        self.num_params = 7
        
        self.jacob_pattern = np.array([[1, 1],
                                      [1, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[0, 5],      # Dw, interception capacity [mm]
                                    [0, 10],     # Betaw, soil misture distribution parameter [-]
                                    [1, 2000],   # Swmax, soil misture depth [mm]
                                    [0, 1],      # a, surface/groundwater division [-]
                                    [1, 120],    # th, time delay for routing [d]
                                    [0, 4],      # c, capillary rise [mm/d]
                                    [0, 1]])     # kw, base flow time parameter [d-1]
        
        self.store_names = ["S1", "S2"]  # Names for the stores
        self.flux_names = ["pe", "ei", "ea", "qse", "qses",
                          "qseg", "c", "qhgw", "qhsrf", "qt"]  # Names for the fluxes
        
        self.flux_groups = {"Ea": [2, 3],   # Index or indices of fluxes to add to Actual ET
                           "Q": [10]}       # Index or indices of fluxes to add to Streamflow

        # setting delta_t and theta triggers the function obj.init()
        if delta_t is not None:
            self.delta_t = delta_t
        if theta is not None:
            self.theta = theta

    def init(self):
        """
        INITialisation function
        """
        # parameters
        theta = self.theta
        delta_t = self.delta_t
        th = theta[4]  # Routing delay [d]
        
        # initialise the unit hydrographs and still-to-flow vectors            
        uh = uh_3_half(th, delta_t)
        
        self.uhs = [uh]

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
        dw = theta[0]     # Daily interception [mm]
        betaw = theta[1]  # Soil moisture storage distribution parameter [-]
        swmax = theta[2]  # Maximum soil moisture storage [mm]
        a = theta[3]      # Division parameter for surface and groundwater flow [-]
        c = theta[5]      # Rate of capillary rise [mm/d]
        kh = theta[6]     # Groundwater runoff coefficient [d-1]
        
        # delta_t
        delta_t = self.delta_t
        
        # unit hydrographs and still-to-flow vectors
        uh = self.uhs[0]
        
        # stores
        S1 = S[0]
        S2 = S[1]
        
        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]
        
        # fluxes functions
        flux_pe = interception_2(P, dw)
        flux_ei = P - flux_pe  # tracks 'intercepted' rainfall
        flux_ea = evap_1(S1, Ep, delta_t)
        flux_qse = saturation_2(S1, swmax, betaw, flux_pe)
        flux_qses = split_1(a, flux_qse)
        flux_qseg = split_1(1 - a, flux_qse)
        flux_c = capillary_2(c, S2, delta_t)
        flux_qhgw = baseflow_1(kh, S2)
        flux_qhsrf = route(flux_qses, uh)
        flux_qt = flux_qhsrf + flux_qhgw
        
        # stores ODEs
        dS1 = flux_pe + flux_c - flux_ea - flux_qse
        dS2 = flux_qseg - flux_c - flux_qhgw
        
        # outputs
        dS = np.array([dS1, dS2])
        fluxes = np.array([flux_pe, flux_ei, flux_ea, flux_qse, flux_qses,
                           flux_qseg, flux_c, flux_qhgw, flux_qhsrf, flux_qt])
        
        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep.
        """
        # unit hydrographs and still-to-flow vectors
        uh = self.uhs[0]
        
        # input fluxes to the unit hydrographs 
        fluxes = self.fluxes[self.t, :] 
        flux_qses = fluxes[4]
        
        # update still-to-flow vectors using fluxes at current step and
        # unit hydrographs
        uh = update_uh(uh, flux_qses)
        self.uhs = [uh]
