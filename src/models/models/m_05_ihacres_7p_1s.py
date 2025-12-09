import numpy as np
from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation.evap_12 import evap_12
from pymarrmot.models.flux.split.split_1 import split_1
from pymarrmot.models.flux.saturation.saturation_5 import saturation_5
from pymarrmot.models.unit_hydro.uh_5_half import uh_5_half
from pymarrmot.models.unit_hydro.uh_8_delay import uh_8_delay
from pymarrmot.models.unit_hydro.route import route
from pymarrmot.models.unit_hydro.update_uh import update_uh

class m_05_ihacres_7p_1s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: IHACRES

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model references:
    - Croke, B. F. W., & Jakeman, A. J. (2004). A catchment moisture deficit
    module for the IHACRES rainfall-runoff model. Environmental Modelling and
    Software, 19(1), 1–5. http://doi.org/10.1016/j.envsoft.2003.09.001
    - Littlewood, I. G., Down, K., Parker, J. R., & Post, D. A. (1997). IHACRES
    v1.0 User Guide.
    - Ye, W., Bates, B. C., Viney, N. R., & Sivapalan, M. (1997). Performance
    of conceptual rainfall-runoff models in low-yielding ephemeral
    catchments. Water Resources Research, 33(1), 153–166.
    http://doi.org/doi:10.1029/96WR02840
    """

    def __init__(self):
        super().__init__()
        self.num_stores = 1  # number of model stores
        self.num_fluxes = 7  # number of model fluxes
        self.num_params = 7

        self.jacob_pattern = [1]  # Jacobian matrix of model store ODEs

        self.par_ranges = [
            [1, 2000],    # lp, Wilting point [mm]
            [1, 2000],    # d, Threshold for flow generation [mm]
            [0, 10],      # p, Flow response non-linearity [-]
            [0, 1],       # alpha, Fast/slow flow division [-]
            [1, 700],     # tau_q, Fast flow routing delay [d]
            [1, 700],     # tau_s, Slow flow routing delay [d]
            [0, 119]      # tau_d, flow delay [d]
        ]

        self.store_names = ["S1"]  # Names for the stores
        self.flux_names = ["Ea", "u", "uq", "us", "xq", "xs", "Qt"]  # Names for the fluxes

        self.flux_groups = {
            "Ea": 1,    # Index or indices of fluxes to add to Actual ET
            "Q": 7      # Index or indices of fluxes to add to Streamflow
        }

        self.StoreSigns = -1  # Signs to give to stores (-1 is a deficit store), only needed for water balance

    def init(self):
        theta = self.theta
        delta_t = self.delta_t

        tau_q = theta[4]  # Fast flow routing delay [d]
        tau_s = theta[5]  # Slow flow routing delay [d]
        tau_d = theta[6]  # Pure time delay of total flow [d]

        # initialise the unit hydrographs and still-to-flow vectors
        uh_q = uh_5_half(tau_q, delta_t)
        uh_s = uh_5_half(tau_s, delta_t)
        uh_t = uh_8_delay(tau_d, delta_t)

        self.uhs = [uh_q, uh_s, uh_t]

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
        lp = theta[0]  # Wilting point [mm]
        d = theta[1]  # Threshold for flow generation [mm]
        p = theta[2]  # Flow response non-linearity [-]
        alpha = theta[3]  # Fast/slow flow division [-]

        # delta_t
        delta_t = self.delta_t

        # unit hydrographs and still-to-flow vectors
        uhs = self.uhs
        uh_q = uhs[0]
        uh_s = uhs[1]
        uh_t = uhs[2]

        # stores
        S1 = S[0]

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        # fluxes functions
        flux_ea = evap_12(S1, lp, Ep)
        flux_u = saturation_5(S1, d, p, P)
        flux_uq = split_1(alpha, flux_u)
        flux_us = split_1(1 - alpha, flux_u)
        flux_xq = route(flux_uq, uh_q)
        flux_xs = route(flux_us, uh_s)
        flux_xt = route(flux_xq + flux_xs, uh_t)

        # stores ODEs
        dS1 = -P + flux_ea + flux_u

        # outputs
        dS = np.array([dS1])
        fluxes = [flux_ea, flux_u, flux_uq, flux_us, flux_xq, flux_xs, flux_xt]

        return dS, fluxes

    def step(self):
        # unit hydrographs and still-to-flow vectors
        uhs = self.uhs
        uh_q = uhs[0]
        uh_s = uhs[1]
        uh_t = uhs[2]

        # input fluxes to the unit hydrographs
        fluxes = self.fluxes[self.t, :]
        flux_uq = fluxes[2]
        flux_us = fluxes[3]
        flux_xq = fluxes[4]
        flux_xs = fluxes[5]

        # update still-to-flow vectors using fluxes at current step and
        # unit hydrographs
        uh_q = update_uh(uh_q, flux_uq)
        uh_s = update_uh(uh_s, flux_us)
        uh_t = update_uh(uh_t, flux_xq + flux_xs)

        self.uhs = [uh_q, uh_s, uh_t]
