import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation import evap_13, evap_14
from pymarrmot.models.flux.split import split_1
from pymarrmot.models.flux.baseflow import baseflow_1
from pymarrmot.models.flux.effective_1 import effective_1
from pymarrmot.models.flux.saturation import saturation_6, saturation_1
from pymarrmot.models.flux.infiltration import infiltration_4
from pymarrmot.models.unit_hydro import (uh_6_gamma, update_uh, route)

class m_40_smar_8p_6s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: SMAR

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model reference:
    O’Connell, P. E., Nash, J. E., & Farrell, J. P. (1970). River flow
    forecasting through conceptual models part II - the Brosna catchment at
    Ferbane. Journal of Hydrology, 10, 317–329.

    Tan, B. Q., & O’Connor, K. M. (1996). Application of an empirical
    infiltration equation in the SMAR conceptual model. Journal of Hydrology,
    185(1-4), 275–295. http://doi.org/10.1016/0022-1694(95)02993-1
    """

    def __init__(self):
        super().__init__()
        self.num_stores = 6  # number of model stores
        self.num_fluxes = 20  # number of model fluxes
        self.num_params = 8

        self.jacob_pattern = np.array([
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1]
        ])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([
            [0, 1],  # h, Maximum fraction of direct runoff [-]
            [0, 200],  # y, Infiltration rate [mm/d]
            [1, 2000],  # smax, Maximum soil moisture storage [mm]
            [0, 1],  # c, Evaporation reduction coefficient [-]
            [0, 1],  # g, Groundwater recharge coefficient [-]
            [0, 1],  # kg, Groundwater time parameter [d-1]
            [1, 10],  # n, Number of Nash cascade reservoirs [-]
            [1, 120]  # n*k, Routing delay [d]
        ])

        self.store_names = ["S1", "S2", "S3", "S4", "S5", "S6"]  # Names for the stores
        self.flux_names = ["pstar", "estar", "evap", "r1", "i",
                          "r2", "e1", "e2", "e3", "e4",
                          "e5", "q1", "q2", "q3", "q4",
                          "r3", "rg", "r3star", "qr", "qg"]  # Names for the fluxes

        self.flux_groups = {'Ea': [3, 7, 8, 9, 10, 11],  # Index or indices of fluxes to add to Actual ET
                           'Q': [19, 20]}  # Index or indices of fluxes to add to Streamflow

    def init(self):
        """
        INITialisation function
        """
        theta = self.theta
        delta_t = self.delta_t
        n = theta[6]  # Number of Nash cascade reservoirs [-]
        nk = theta[7]  # Routing delay [d]. n and k are optimized together
        k = nk / n  # time parameter in the gamma function

        # initialise the unit hydrographs and still-to-flow vectors
        uh = uh_6_gamma(n, k, delta_t)

        self.uhs = [uh]

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation
        """
        theta = self.theta
        h = theta[0]  # Maximum fraction of direct runoff [-]
        y = theta[1]  # Infiltration rate [mm/d]
        smax = theta[2]  # Maximum soil moisture storage [mm]
        c = theta[3]  # Evaporation reduction coefficient [-]
        g = theta[4]  # Groundwater recharge coefficient [-]
        kg = theta[5]  # Groundwater time parameter [d-1]

        delta_t = self.delta_t

        uhs = self.uhs
        uh = uhs[0]

        S1, S2, S3, S4, S5, S6 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        flux_pstar = effective_1(P, Ep)
        flux_estar = effective_1(Ep, P)
        flux_evap = min(Ep, P)
        flux_r1 = saturation_6(h, (S1 + S2 + S3 + S4 + S5), smax, flux_pstar)
        flux_i = infiltration_4(flux_pstar - flux_r1, y)
        flux_r2 = effective_1(flux_pstar - flux_r1, flux_i)
        flux_e1 = evap_13(c, 0, flux_estar, S1, delta_t)
        flux_e2 = evap_14(c, 1, flux_estar, S2, S1, 0.1, delta_t)
        flux_e3 = evap_14(c, 2, flux_estar, S3, S2, 0.1, delta_t)
        flux_e4 = evap_14(c, 3, flux_estar, S4, S3, 0.1, delta_t)
        flux_e5 = evap_14(c, 4, flux_estar, S5, S4, 0.1, delta_t)
        flux_q1 = saturation_1(flux_i, S1, smax / 5)
        flux_q2 = saturation_1(flux_q1, S2, smax / 5)
        flux_q3 = saturation_1(flux_q2, S3, smax / 5)
        flux_q4 = saturation_1(flux_q3, S4, smax / 5)
        flux_r3 = saturation_1(flux_q4, S5, smax / 5)
        flux_rg = split_1(g, flux_r3)
        flux_r3star = split_1(1 - g, flux_r3)
        flux_qr = route(flux_r1 + flux_r2 + flux_r3star, uh)
        flux_qg = baseflow_1(kg, S6)

        # stores ODEs
        dS1 = flux_i - flux_e1 - flux_q1
        dS2 = flux_q1 - flux_e2 - flux_q2
        dS3 = flux_q2 - flux_e3 - flux_q3
        dS4 = flux_q3 - flux_e4 - flux_q4
        dS5 = flux_q4 - flux_e5 - flux_r3
        dS6 = flux_rg - flux_qg

        # outputs
        dS = np.array([dS1, dS2, dS3, dS4, dS5, dS6])
        fluxes = [flux_pstar, flux_estar, flux_evap, flux_r1, flux_i,
                  flux_r2, flux_e1, flux_e2, flux_e3, flux_e4,
                  flux_e5, flux_q1, flux_q2, flux_q3, flux_q4,
                  flux_r3, flux_rg, flux_r3star, flux_qr, flux_qg]

        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep.
        """
        uhs = self.uhs
        uh = uhs[0]

        fluxes = self.fluxes[self.t, :]
        flux_r1 = fluxes[3]
        flux_r2 = fluxes[5]
        flux_r3star = fluxes[17]

        uh = update_uh(uh, flux_r1 + flux_r2 + flux_r3star)
        self.uhs = [uh]