import numpy as np
from models.marrmot_model import MARRMoT_model
from models.flux import (evap_7, saturation_1, recharge_3, baseflow_1)

class m_24_mopex1_5p_4s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: MOPEX-1

    Model reference:
    Ye, S., Yaeger, M., Coopersmith, E., Cheng, L., & Sivapalan, M. (2012).
    Exploring the physical controls of regional patterns of flow duration
    curves - Part 2: Role of seasonality, the regime curve, and associated
    process controls. Hydrology and Earth System Sciences, 16(11), 4447–4465.
    http://doi.org/10.5194/hess-16-4447-2012
    """
    def __init__(self):
        super().__init__()
        self.numStores = 4                                          # number of model stores
        self.numFluxes = 7                                          # number of model fluxes
        self.numParams = 5

        self.JacobPattern = np.array([[1, 0, 0, 0],
                                       [1, 1, 0, 0],
                                       [1, 0, 1, 0],
                                       [0, 1, 0, 1]])              # Jacobian matrix of model store ODEs

        self.parRanges = np.array([[1, 2000],                       # Sb1, Maximum soil moisture storage [mm]
                                    [0, 1],                          # tw, Groundwater leakage time [d-1]
                                    [0, 1],                          # tu, Slow flow routing response time [d-1]
                                    [1, 2000],                       # se, Root zone storage capacity [mm]
                                    [0, 1]])                         # tc, Mean residence time [d-1]

        self.StoreNames = ["S1", "S2", "S3", "S4"]                   # Names for the stores
        self.FluxNames = ["et1", "q1f", "qw",
                          "et2", "q2u", "qf", "qs"]                # Names for the fluxes

        self.FluxGroups = {"Ea": [1, 4],                            # Index or indices of fluxes to add to Actual ET
                           "Q": [6, 7]}                            # Index or indices of fluxes to add to Streamflow

    def init(self):
        """
        Initialization function.
        """
        pass

    def model_fun(self, S):
        """
        Model governing equations in state-space formulation.

        Parameters:
        -----------
        S : numpy.ndarray
            State variables.

        Returns:
        --------
        tuple
            State derivatives and fluxes.
        """
        # parameters
        s1max, tw, tu, se, tc = self.theta

        # delta_t
        delta_t = self.delta_t

        # stores
        S1, S2, S3, S4 = S

        # climate input
        t = self.t                                  # this time step
        climate_in = self.input_climate[t, :]       # climate at this step
        P, Ep, _ = climate_in

        # fluxes functions
        flux_et1 = evap_7(S1, s1max, Ep, delta_t)
        flux_q1f = saturation_1(P, S1, s1max)
        flux_qw = recharge_3(tw, S1)
        flux_et2 = evap_7(S2, se, Ep, delta_t)
        flux_q2u = baseflow_1(tu, S2)
        flux_qf = baseflow_1(tc, S3)
        flux_qs = baseflow_1(tc, S4)

        # stores ODEs
        dS1 = P - flux_et1 - flux_q1f - flux_qw
        dS2 = flux_qw - flux_et2 - flux_q2u
        dS3 = flux_q1f - flux_qf
        dS4 = flux_q2u - flux_qs

        # outputs
        dS = np.array([dS1, dS2, dS3, dS4])
        fluxes = np.array([flux_et1, flux_q1f, flux_qw,
                           flux_et2, flux_q2u, flux_qf, flux_qs])

        return dS, fluxes

    def step(self):
        """
        Runs at the end of every timestep.
        """
        pass
