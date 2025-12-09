import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.interception import interception_2
from pymarrmot.models.flux.infiltration import infiltration_4
from pymarrmot.models.flux.evaporation import evap_4
from pymarrmot.models.flux.capillary import capillary_2
from pymarrmot.models.flux.saturation import saturation_1
from pymarrmot.models.flux.baseflow import baseflow_1
from pymarrmot.models.unit_hydro.route import route
from pymarrmot.models.unit_hydro.uh_3_half import uh_3_half
from pymarrmot.models.unit_hydro.update_uh import update_uh

class m_15_plateau_8p_2s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Plateau model

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model reference
    Savenije, H. H. G. (2010). “Topography driven conceptual modelling
    (FLEX-Topo).” Hydrology and Earth System Sciences, 14(12), 2681–2692.
    https://doi.org/10.5194/hess-14-2681-2010
    """

    def __init__(self):
        """
        creator method
        """
        super().__init__()
        self.num_stores = 2  # number of model stores
        self.num_fluxes = 9  # number of model fluxes
        self.num_params = 8

        self.jacob_pattern = np.array([[1, 1],
                                      [1, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[0, 200],    # Fmax, maximum infiltration rate [mm/d]
                                   [0, 5],      # Dp, interception capacity [mm]
                                   [1, 2000],   # SUmax, soil misture depth [mm]
                                   [0.05, 0.95],  # Swp, wilting point as fraction of Sumax [-]
                                   [0, 1],      # p, coefficient for moisture constrained evaporation [-]
                                   [1, 120],    # tp, time delay for routing [d]
                                   [0, 4],      # c, capillary rise [mm/d]
                                   [0, 1]])     # kp, base flow time parameter [d-1]

        self.store_names = ["S1", "S2"]  # Names for the stores
        self.flux_names = ["pe", "ei", "pie", "pi",
                          "et", "r", "c", "qpgw", "qpieo"]  # Names for the fluxes

        self.flux_groups = {"Ea": [2, 5],  # Index or indices of fluxes to add to Actual ET
                           "Q": [8, 9]}    # Index or indices of fluxes to add to Streamflow

    def init(self):
        """
        INITialisation function
        """
        #super().__init__()
        # parameters
        theta = self.theta
        delta_t = self.delta_t
        tp = theta[5]  # Time delay of surface flow [d]

        # min and max of stores
        self.store_min = np.zeros(self.num_stores)
        self.store_max = np.full(self.num_stores, np.inf)

        # initialise the unit hydrographs and still-to-flow vectors
        uh = uh_3_half(tp, delta_t)

        self.uhs = [uh]

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation
        """
        # parameters
        theta = self.theta
        fmax = theta[0]  # Maximum infiltration rate [mm/d]
        dp = theta[1]    # Daily interception [mm]
        sumax = theta[2]  # Maximum soil moisture storage [mm]
        lp = theta[3]    # Wilting point [-], defined as lp*sumax
        p = theta[4]     # Parameter for moisture constrained evaporation [-]
        tp = theta[5]    # Time delay of surface flow [d]
        c = theta[6]     # Rate of capillary rise [mm/d]
        kp = theta[7]    # Groundwater runoff coefficient [d-1]

        # delta_t
        delta_t = self.delta_t

        # unit hydrographs and still-to-flow vectors
        uhs = self.uhs
        uh = uhs[0]

        # stores
        S1, S2 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]
        
        # fluxes functions
        flux_pe = interception_2(P, dp)
        flux_ei = P - flux_pe  # track 'intercepted' water
        flux_pi = infiltration_4(flux_pe, fmax)
        flux_pie = flux_pe - flux_pi
        flux_et = evap_4(Ep, p, S1, lp, sumax, delta_t)
        flux_c = capillary_2(c, S2, delta_t)
        flux_r = saturation_1((flux_pi + flux_c), S1, sumax)
        flux_qpgw = baseflow_1(kp, S2)
        flux_qpieo = route(flux_pie, uh)

        # stores ODEs
        dS1 = flux_pi + flux_c - flux_et - flux_r
        dS2 = flux_r - flux_c - flux_qpgw

        # outputs
        dS = np.array([dS1, dS2])
        fluxes = np.array([flux_pe, flux_ei, flux_pie, flux_pi,
                           flux_et, flux_r, flux_c, flux_qpgw, flux_qpieo])

        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep
        """
        # unit hydrographs and still-to-flow vectors
        uhs = self.uhs
        uh = uhs[0]

        # input fluxes to the unit hydrographs
        fluxes = self.fluxes[self.t]
        flux_pie = fluxes[2]

        # update still-to-flow vectors using fluxes at current step and
        # unit hydrographs
        uh = update_uh(uh, flux_pie)
        self.uhs = [uh]
