import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.interception import interception_1
from pymarrmot.models.flux.saturation import saturation_10, saturation_1
from pymarrmot.models.flux.evaporation import evap_17, evap_1
from pymarrmot.models.flux.effective_1 import effective_1
from pymarrmot.models.flux.interflow import interflow_9
from pymarrmot.models.flux.percolation import percolation_6
from pymarrmot.models.flux.baseflow import baseflow_7
from pymarrmot.models.flux.routing_1 import routing_1
from pymarrmot.models.unit_hydro import (uh_7_uniform, update_uh, route)

class m_39_mcrm_16p_5s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Midland Catchment Runoff Model

    References:
    Moore, R. J., & Bell, V. A. (2001). Comparison of rainfall-runoff models
    for flood forecasting. Part 1: Literature review of models. Bristol:
    Environment Agency.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.num_stores = 5  # Number of model stores
        self.num_fluxes = 12  # Number of model fluxes
        self.num_params = 16

        self.jacob_pattern = np.array([[1, 0, 0, 0, 0],
                                       [1, 1, 0, 0, 0],
                                       [0, 1, 1, 0, 0],
                                       [1, 1, 1, 1, 0],
                                       [1, 1, 1, 1, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[0, 5],         # smax, Maximum interception storage [mm]
                                     [0.01, 0.99],  # cmax, Maximum fraction of area contributing to rapid runoff [-]
                                     [0.01, 0.99],  # ct, Fraction of cmax that is the minimum contributing area [-]
                                     [0, 2],        # c1, Shape parameter for rapid flow distribution [mm-1]
                                     [0, 1],        # ce, Shape parameter for evaporation [mm-1]
                                     [1, 2000],     # dsurp, Threshold for direct runoff [mm]
                                     [0, 1],        # kd, Direct runoff time parameter [d-1]
                                     [1, 5],        # gamd, Direct runoff flow non-linearity [-]
                                     [0, 20],       # qpmax, Maximum percolation rate [mm/d]
                                     [0, 1],        # kg, Groundwater time parameter [d-1]
                                     [1, 120],      # tau, Routing delay [d]
                                     [1, 300],      # sbf, Maximum routing store depth [mm]
                                     [0, 1],        # kcr, Channel flow time parameter [d-1]
                                     [1, 5],        # gamcr, Channel flow non-linearity [-]
                                     [0, 1],        # kor, Out-of-bank flow time parameter [d-1]
                                     [1, 5]])       # gamor, Out-of-bank flow non-linearity [-]

        self.store_names = ["S1", "S2", "S3", "S4", "S5"]           # Names for the stores
        self.flux_names = ["ec", "qt", "qr", "er", "qn", "qd",      # Names for the fluxes
                           "qp", "qb", "uib", "uob", "qic", "qoc"]  

        self.flux_groups = {"Ea": [1, 4],   # Index or indices of fluxes to add to Actual ET
                            "Q": [11, 12]}  # Index or indices of fluxes to add to Streamflow

    def init(self):
        """
        Initialization function.
        """
        theta = self.theta
        delta_t = self.delta_t
        cmax = theta[1]     # Maximum fraction of area contributing to rapid runoff [-]
        ct = theta[2]       # Fraction of cmax that is the minimum contributing area c0 [-]
        tau = theta[10]     # Routing delay [d]

        c0 = ct * cmax      # Minimum fraction of area contributing to rapid runoff [-]
        self.aux_theta = [c0]

        self.store_min = np.array([0, -1E6, 0, 0, 0])  # Min and max of stores
        self.store_max = np.array([np.inf] * self.num_stores)

        uh = uh_7_uniform(tau, delta_t)  # Initialize the unit hydrographs and still-to-flow vectors
        self.uhs = [uh]

    def model_fun(self, S):
        """
        Model governing equations in state-space formulation.

        Parameters:
        S (array): State variables

        Returns:
        dS (array): Derivatives of state variables
        fluxes (array): Model fluxes
        """
        theta = self.theta
        smax = theta[0]     # Maximum interception storage [mm]
        cmax = theta[1]     # Maximum fraction of area contributing to rapid runoff [-]
        c1 = theta[3]       # Shape parameter for rapid flow distribution [mm-1]
        ce = theta[4]       # Shape parameter for evaporation [mm-1]
        dsurp = theta[5]    # Threshold for direct runoff [mm]
        kd = theta[6]       # Direct runoff time parameter [d-1]
        gamd = theta[7]     # Direct runoff flow non-linearity [-]
        qpmax = theta[8]    # Maximum percolation rate [mm/d]
        kg = theta[9]       # Groundwater time parameter [d-1]
        sbf = theta[11]     # Maximum routing store depth [mm]
        kcr = theta[12]     # Channel flow time parameter [d-1]
        gamcr = theta[13]   # Channel flow non-linearity [-]
        kor = theta[14]     # Out-of-bank flow time parameter [d-1]
        gamor = theta[15]   # Out-of-bank flow non-linearity [-]

        c0 = self.aux_theta[0]

        delta_t = self.delta_t

        uhs = self.uhs
        uh = uhs[0]

        S1, S2, S3, S4, S5 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]
              
        flux_ec = evap_1(S1, Ep, delta_t)
        flux_qt = interception_1(P, S1, smax)
        flux_qr = saturation_10(cmax, c0, c1, S2, flux_qt)
        flux_er = evap_17(ce, S2, Ep - flux_ec)
        flux_qn = effective_1(flux_qt, flux_qr)
        flux_qd = interflow_9(S2, kd, dsurp, gamd, delta_t)
        flux_qp = percolation_6(qpmax, dsurp, S2, delta_t)
        flux_qb = baseflow_7(kg, 1.5, S3, delta_t)
        flux_uib = route(flux_qr + flux_qd + flux_qb, uh)
        flux_uob = saturation_1(flux_uib, S4, sbf)
        flux_qic = routing_1(kcr, gamcr, 3 / 4, S4, delta_t)
        flux_qoc = routing_1(kor, gamor, 3 / 4, S5, delta_t)

        dS1 = P - flux_ec - flux_qt
        dS2 = flux_qn - flux_er - flux_qd - flux_qp
        dS3 = flux_qp - flux_qb
        dS4 = flux_uib - flux_uob - flux_qic
        dS5 = flux_uob - flux_qoc

        dS = np.array([dS1, dS2, dS3, dS4, dS5])
        fluxes = [flux_ec, flux_qt, flux_qr, flux_er,
                  flux_qn, flux_qd, flux_qp, flux_qb,
                  flux_uib, flux_uob, flux_qic, flux_qoc]

        return dS, fluxes

    def step(self):
        """
        Runs at the end of every timestep, used to update still-to-flow vectors from unit hydrographs.
        """
        uhs = self.uhs
        uh = uhs[0]

        fluxes = self.fluxes[self.t]
        flux_qr = fluxes[2]
        flux_qd = fluxes[5]
        flux_qb = fluxes[7]

        uh = update_uh(uh, flux_qr + flux_qd + flux_qb)
        self.uhs = [uh]

