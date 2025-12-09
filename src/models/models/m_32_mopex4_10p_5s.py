import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.snowfall import snowfall_1
from pymarrmot.models.flux.rainfall import rainfall_1
from pymarrmot.models.flux.melt import melt_1
from pymarrmot.models.flux.evaporation import evap_7
from pymarrmot.models.flux.interception import interception_4
from pymarrmot.models.flux.saturation import saturation_1
from pymarrmot.models.flux.recharge import recharge_3
from pymarrmot.models.flux.baseflow import baseflow_1

class m_32_mopex4_10p_5s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: MOPEX-4

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
        self.num_fluxes = 12  # number of model fluxes
        self.num_params = 10

        self.jacob_pattern = [[1, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0],
                              [0, 1, 1, 0, 0],
                              [0, 1, 1, 1, 0],
                              [0, 0, 1, 0, 1]]  # Jacobian matrix of model store ODEs

        self.par_ranges = [[-3, 3],        # tcrit, Snowfall & snowmelt temperature [oC]
                           [0, 20],        # ddf, Degree-day factor for snowmelt [mm/oC/d]
                           [1, 2000],      # Sb1, Maximum soil moisture storage [mm]
                           [0, 1],         # tw, Groundwater leakage time [d-1]
                           [0, 1],         # I_alpha, Intercepted fraction of Pr [-]
                           [1, 365],       # I_s, Maximum Leaf Area Index timing [d]
                           [0, 1],         # tu, Slow flow routing response time [d-1]
                           [0.05, 0.95],   # se, Root zone storage capacity as fraction of Sb2 [-]
                           [1, 2000],      # Sb2, Root zone storage capacity [mm]
                           [0, 1]]         # tc, Mean residence time [d-1]

        self.store_names = ["S1", "S2", "S3", "S4", "S5"]  # Names for the stores
        self.flux_names = ["ps", "pr", "qn", "et1", "i", "q1f",
                           "qw", "et2", "q2f", "q2u", "qf", "qs"]  # Names for the fluxes

        self.flux_groups = {"Ea": [4, 5, 8],  # Index or indices of fluxes to add to Actual ET
                            "Q": [11, 12]}    # Index or indices of fluxes to add to Streamflow

    def init(self):
        pass

    def model_fun(self, S):
        """
        Model governing equations in state-space formulation
        """
        theta = self.theta
        tcrit = theta[0]  # Snowfall & snowmelt temperature [oC]
        ddf = theta[1]    # Degree-day factor for snowmelt [mm/oC/d]
        s2max = theta[2]  # Maximum soil moisture storage [mm]
        tw = theta[3]     # Groundwater leakage time [d-1]
        i_alpha = theta[4]  # Intercepted fraction of Pr [-]
        i_s = theta[5]      # Maximum Leaf Area Index timing [d]
        tu = theta[6]       # Slow flow routing response time [d-1]
        se = theta[7]       # Root zone storage capacity as fraction of s3max [-]
        s3max = theta[8]    # Maximum groundwater storage [mm]
        tc = theta[9]       # Mean residence time [d-1]

        # Auxiliary parameters
        tmax = 365.25  # Duration of seasonal cycle [d]

        # Delta_t
        delta_t = self.delta_t

        # Stores
        S1, S2, S3, S4, S5 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        # Fluxes functions
        flux_ps = snowfall_1(P, T, tcrit)
        flux_pr = rainfall_1(P, T, tcrit)
        flux_qn = melt_1(ddf, tcrit, T, S1, delta_t)
        flux_et1 = evap_7(S2, s2max, Ep, delta_t)
        flux_i = interception_4(i_alpha, i_s, self.t+1, tmax, flux_pr, delta_t)
        flux_q1f = saturation_1(flux_pr + flux_qn, S2, s2max)
        flux_qw = recharge_3(tw, S2)
        flux_et2 = evap_7(S3, se * s3max, Ep, delta_t)
        flux_q2f = saturation_1(flux_qw, S3, s3max)
        flux_q2u = baseflow_1(tu, S3)
        flux_qf = baseflow_1(tc, S4)
        flux_qs = baseflow_1(tc, S5)

        # Stores ODEs
        dS1 = flux_ps - flux_qn
        dS2 = flux_pr + flux_qn - flux_et1 - flux_i - flux_q1f - flux_qw
        dS3 = flux_qw - flux_et2 - flux_q2f - flux_q2u
        dS4 = flux_q1f + flux_q2f - flux_qf
        dS5 = flux_q2u - flux_qs

        # Outputs
        dS = np.array([dS1, dS2, dS3, dS4, dS5])
        fluxes = [flux_ps, flux_pr, flux_qn, flux_et1,
                  flux_i, flux_q1f, flux_qw, flux_et2,
                  flux_q2f, flux_q2u, flux_qf, flux_qs]
        return dS, fluxes

    def step(self):
        pass
