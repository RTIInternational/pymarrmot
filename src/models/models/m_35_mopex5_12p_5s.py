import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.phenology import phenology_1
from pymarrmot.models.flux.snowfall import snowfall_1
from pymarrmot.models.flux.rainfall import rainfall_1
from pymarrmot.models.flux.melt import melt_1
from pymarrmot.models.flux.evaporation import evap_7
from pymarrmot.models.flux.interception import interception_4
from pymarrmot.models.flux.saturation import saturation_1
from pymarrmot.models.flux.recharge import recharge_3
from pymarrmot.models.flux.baseflow import baseflow_1

class m_35_mopex5_12p_5s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: MOPEX-5

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model reference:
    Ye, S., Yaeger, M., Coopersmith, E., Cheng, L., & Sivapalan, M. (2012).
    Exploring the physical controls of regional patterns of flow duration
    curves - Part 2: Role of seasonality, the regime curve, and associated
    process controls. Hydrology and Earth System Sciences, 16(11), 4447–4465.
    http://doi.org/10.5194/hess-16-4447-2012
    """

    def __init__(self):
        super().__init__()
        self.num_stores = 5  # number of model stores
        self.num_fluxes = 13  # number of model fluxes
        self.num_params = 12

        # Jacobian matrix of model store ODEs
        self.jacob_pattern = np.array([[1, 0, 0, 0, 0],
                                       [1, 1, 0, 0, 0],
                                       [0, 1, 1, 0, 0],
                                       [0, 1, 1, 1, 0],
                                       [0, 0, 1, 0, 1]])

        # Parameter ranges
        self.par_ranges = np.array([[-3, 3],       # tcrit
                                    [0, 20],        # ddf
                                    [1, 2000],      # Sb1
                                    [0, 1],         # tw
                                    [0, 1],         # I_alpha
                                    [1, 365],       # I_s
                                    [-10, 0],       # tmin
                                    [1, 20],        # trange
                                    [0, 1],         # tu
                                    [0.05, 0.95],   # se
                                    [1, 2000],      # Sb2
                                    [0, 1]])        # tc

        # Names for the stores and fluxes
        self.store_names = ["S1", "S2", "S3", "S4", "S5"]
        self.flux_names = ["epc", "ps", "pr", "qn", "et1", "i", "q1f", "qw",
                           "et2", "q2f", "q2u", "qf", "qs"]

        # Flux groups
        self.flux_groups = {"Ea": [5, 6, 9], "Q": [12, 13]}
        #self.flux_groups["Ea"] = [5, 6, 9]
        #self.flux_groups["Q"] = [12, 13]

    def init(self):
        """Initialize the MOPEX-5 model."""
        self.store_min = np.zeros(self.num_stores)
        self.store_max = np.full(self.num_stores, np.inf)

    def model_fun(self, S):
        """
        Model governing equations in state-space formulation.

        Parameters:
        -----------
        S : array_like
            State variables.

        Returns:
        --------
        dS : array_like
            Time derivatives of the state variables.
        fluxes : array_like
            Fluxes.

        """
        theta = self.theta
        tcrit, ddf, s2max, tw, i_alpha, i_s, tmin, trange, tu, se, s3max, tc = theta

        tmax = 365.25  # Duration of seasonal cycle [d]
        delta_t = self.delta_t

        S1, S2, S3, S4, S5 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        flux_epc = phenology_1(T, tmin, tmin + trange, Ep)
        flux_ps = snowfall_1(P, T, tcrit)
        flux_pr = rainfall_1(P, T, tcrit)
        flux_qn = melt_1(ddf, tcrit, T, S1, delta_t)
        flux_et1 = evap_7(S2, s2max, flux_epc, delta_t)
        flux_i = interception_4(i_alpha, i_s, self.t+1, tmax, flux_pr, delta_t)
        flux_q1f = saturation_1(flux_pr + flux_qn, S2, s2max)
        flux_qw = recharge_3(tw, S2)
        flux_et2 = evap_7(S3, se * s3max, flux_epc, delta_t)
        flux_q2f = saturation_1(flux_qw, S3, s3max)
        flux_q2u = baseflow_1(tu, S3)
        flux_qf = baseflow_1(tc, S4)
        flux_qs = baseflow_1(tc, S5)

        dS1 = flux_ps - flux_qn
        dS2 = flux_pr + flux_qn - flux_et1 - flux_i - flux_q1f - flux_qw
        dS3 = flux_qw - flux_et2 - flux_q2f - flux_q2u
        dS4 = flux_q1f + flux_q2f - flux_qf
        dS5 = flux_q2u - flux_qs

        dS = np.array([dS1, dS2, dS3, dS4, dS5])
        fluxes = np.array([flux_epc, flux_ps, flux_pr, flux_qn, flux_et1,
                           flux_i, flux_q1f, flux_qw, flux_et2, flux_q2f,
                           flux_q2u, flux_qf, flux_qs])

        return dS, fluxes

    def step(self):
        """Run at the end of every timestep."""
        pass
