import numpy as np
from marrmot_model import MARRMoT_model
from Models.Flux import (evap_1, effective_1, split_1, evap_16,
                         saturation_1, saturation_9, baseflow_1, abstraction_1, baseflow_6)

class m_25_tcm_6p_4s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Thames Catchment Model

    Model reference:
    Moore, R. J., & Bell, V. A. (2001). Comparison of rainfall-runoff models
    for flood forecasting. Part 1: Literature review of models. Bristol:
    Environment Agency.
    """
    def __init__(self):
        super().__init__()
        self.numStores = 4                                          # number of model stores
        self.numFluxes = 11                                         # number of model fluxes
        self.numParams = 6

        self.JacobPattern = np.array([[1, 0, 0, 0],
                                       [1, 1, 0, 0],
                                       [1, 1, 1, 0],
                                       [0, 0, 1, 1]])              # Jacobian matrix of model store ODEs

        self.parRanges = np.array([[0, 1],                           # phi, Fraction preferential recharge [-]
                                    [1, 2000],                       # rc, Maximum soil moisture depth [mm]
                                    [0, 1],                          # gam, Fraction of Ep reduction with depth [-]
                                    [0, 1],                          # k1, Runoff coefficient [d-1]
                                    [0, 1],                          # fa, Fraction of mean(P) that forms abstraction rate [mm/d]
                                    [0, 1]])                         # k2, Runoff coefficient [mm-1 d-1]

        self.StoreNames = ["S1", "S2", "S3", "S4"]                   # Names for the stores
        self.FluxNames = ["pn", "en", "pby", "pin", "ea",
                          "et", "qex1", "qex2", "quz", "a", "q"]    # Names for the fluxes

        self.FluxGroups = {"Ea": [2, 5, 6],                          # Index or indices of fluxes to add to Actual ET
                           "Q": 11,                                 # Index or indices of fluxes to add to Streamflow
                           "Abstraction": 10}                      # Index or abstraction flux (just needed for water balance)

        self.StoreSigns = [1, -1, 1, 1]                              # Signs to give to stores (-1 is a deficit store), only needed for water balance

    def init(self):
        """
        Initialization function.
        """
        fa = self.theta[4]    # Fraction of average P abstracted [-]
        P = self.input_climate[:, 0]

        ca = fa * np.mean(P)    # Abstraction rate [mm/day]
        self.aux_theta = np.array([ca])

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
        phi, rc, gam, k1, _, k2 = self.theta
        ca = self.aux_theta[0]    # Abstraction rate [mm/day]

        # delta_t
        delta_t = self.delta_t

        # stores
        S1, S2, S3, S4 = S

        # climate input
        t = self.t                                 # this time step
        climate_in = self.input_climate[t, :]      # climate at this step
        P, Ep, _ = climate_in

        # fluxes functions
        flux_pn = effective_1(P, Ep)
        flux_en = P - flux_pn
        flux_pby = split_1(phi, flux_pn)
        flux_pin = split_1(1 - phi, flux_pn)
        flux_ea = evap_1(S1, Ep, delta_t)
        flux_et = evap_16(gam, np.inf, S1, 0.01, Ep, delta_t)
        flux_qex1 = saturation_1(flux_pin, S1, rc)
        flux_qex2 = saturation_9(flux_qex1, S2, 0.01)
        flux_quz = baseflow_1(k1, S3)
        flux_a = abstraction_1(ca)
        flux_q = baseflow_6(k2, 0, S4, delta_t)

        # stores ODEs
        dS1 = flux_pin - flux_ea - flux_qex1
        dS2 = flux_et + flux_qex2 - flux_qex1
        dS3 = flux_qex2 + flux_pby - flux_quz
        dS4 = flux_quz - flux_a - flux_q

        # outputs
        dS = np.array([dS1, dS2, dS3, dS4])
        fluxes = np.array([flux_pn, flux_en, flux_pby, flux_pin,
                           flux_ea, flux_et, flux_qex1, flux_qex2,
                           flux_quz, flux_a, flux_q])

        return dS, fluxes

    def step(self):
        """
        Runs at the end of every timestep.
        """
        pass
