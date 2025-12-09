import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.snowfall import snowfall_1
from pymarrmot.models.flux.rainfall import rainfall_1
from pymarrmot.models.flux.melt import melt_1
from pymarrmot.models.flux.evaporation import evap_7
from pymarrmot.models.flux.saturation import saturation_1
from pymarrmot.models.flux.recharge import recharge_3
from pymarrmot.models.flux.baseflow import baseflow_1

class m_30_mopex2_7p_5s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: MOPEX-2

    References:
    - Ye, S., Yaeger, M., Coopersmith, E., Cheng, L., & Sivapalan, M. (2012).
      Exploring the physical controls of regional patterns of flow duration
      curves - Part 2: Role of seasonality, the regime curve, and associated
      process controls. Hydrology and Earth System Sciences, 16(11), 4447–4465.
      http://doi.org/10.5194/hess-16-4447-2012
    """

    def __init__(self):
        super().__init__()
        self.num_stores = 5  # number of model stores
        self.num_fluxes = 10  # number of model fluxes
        self.num_params = 7

        self.jacob_pattern = np.array([[1, 0, 0, 0, 0],
                                       [1, 1, 0, 0, 0],
                                       [0, 1, 1, 0, 0],
                                       [0, 1, 0, 1, 0],
                                       [0, 0, 1, 0, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[-3, 3],     # tcrit, Snowfall & snowmelt temperature [oC]
                                     [0, 20],     # ddf, Degree-day factor for snowmelt [mm/oC/d]
                                     [1, 2000],   # Sb1, Maximum soil moisture storage [mm]
                                     [0, 1],      # tw, Groundwater leakage time [d-1]
                                     [0, 1],      # tu, Slow flow routing response time [d-1]
                                     [1, 2000],   # se, Root zone storage capacity [mm]
                                     [0, 1]])     # tc, Mean residence time [d-1]

        self.store_names = ["S1", "S2", "S3", "S4", "S5"]  # Names for the stores
        self.flux_names = ["ps", "pr", "qn", "et1", "q1f",
                           "qw", "et2", "q2u", "qf", "qs"]  # Names for the fluxes

        self.flux_groups = {"Ea": [4, 7],  # Index or indices of fluxes to add to Actual ET
                            "Q": [9, 10]}   # Index or indices of fluxes to add to Streamflow

    def init(self):
        """
        INITialization function
        """
        pass

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation

        Parameters:
        - S: array-like, current state variables

        Returns:
        - dS: array-like, model state derivatives
        - fluxes: array-like, model fluxes
        """
        # parameters
        tcrit, ddf, s2max, tw, tu, se, tc = self.theta

        # delta_t
        delta_t = self.delta_t

        # stores
        S1, S2, S3, S4, S5 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]
        
        # fluxes functions
        flux_ps = snowfall_1(P, T, tcrit)
        flux_pr = rainfall_1(P, T, tcrit)
        flux_qn = melt_1(ddf, tcrit, T, S1, delta_t)
        flux_et1 = evap_7(S2, s2max, Ep, delta_t)
        flux_q1f = saturation_1(flux_pr + flux_qn, S2, s2max)
        flux_qw = recharge_3(tw, S2)
        flux_et2 = evap_7(S3, se, Ep, delta_t)
        flux_q2u = baseflow_1(tu, S3)
        flux_qf = baseflow_1(tc, S4)
        flux_qs = baseflow_1(tc, S5)

        # stores ODEs
        dS1 = flux_ps - flux_qn
        dS2 = flux_pr + flux_qn - flux_et1 - flux_q1f - flux_qw
        dS3 = flux_qw - flux_et2 - flux_q2u
        dS4 = flux_q1f - flux_qf
        dS5 = flux_q2u - flux_qs

        # outputs
        dS = np.array([dS1, dS2, dS3, dS4, dS5])
        fluxes = [flux_ps, flux_pr, flux_qn, flux_et1, flux_q1f,
                  flux_qw, flux_et2, flux_q2u, flux_qf, flux_qs]

        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep
        """
        pass
