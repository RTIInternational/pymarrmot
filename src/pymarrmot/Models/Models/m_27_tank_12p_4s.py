import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation import evap_1
from pymarrmot.models.flux.interflow import interflow_8
from pymarrmot.models.flux.baseflow import baseflow_1
from pymarrmot.models.flux.recharge import recharge_3

class m_27_tank_12p_4s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Tank model

    Model reference:
    Sugawara, M. (1995). Tank model. In V. P. Singh (Ed.), Computer models of
    watershed hydrology (pp. 165–214). Water Resources Publications, USA.
    """
    def __init__(self):
        super().__init__()
        self.aux_theta = None

    def __init__(self):
        super().__init__()
        self.numStores = 4                                              # number of model stores
        self.numFluxes = 12                                             # number of model fluxes
        self.numParams = 12

        self.JacobPattern = np.array([[1, 0, 0, 0],
                                      [1, 1, 0, 0],
                                      [1, 1, 1, 0],
                                      [1, 1, 1, 1]])                   # Jacobian matrix of model store ODEs

        self.parRanges = np.array([[0, 1],                              # a0, Time parameter for drainage 1>2 [d-1]
                                    [0, 1],                              # b0, Time parameter for drainage 2>3 [d-1]
                                    [0, 1],                              # c0, Time parameter for drainage 3>4 [d-1]
                                    [0, 1],                              # a1, Time parameter for surface runoff 1 [d-1]
                                    [0, 1],                              # fa, Fraction of a1 that is a2 [-]
                                    [0, 1],                              # fb, Fraction of a2 that is b1 [-]
                                    [0, 1],                              # fc, Fraction of b1 that is c1 [-]
                                    [0, 1],                              # fd, Fraction of c1 that is d1 [-]
                                    [1, 2000],                           # st, Maximum soil depth (sum of runoff thresholds) [mm]
                                    [0.01, 0.99],                        # f2, Fraction of st that consitutes threshold t2 [-]
                                    [0.01, 0.99],                        # f1, Fraction of st-t2 that is added to t2 to find threshold 1 [-] (ensures t1 > t2)
                                    [0.01, 0.99]])                       # f3, Fraction of st-t1-t2 that consitutes threshold 3 [-]

        self.StoreNames = ["S1", "S2", "S3", "S4"]                       # Names for the stores
        self.FluxNames = ["y1", "y2", "y3", "y4", "y5",
                          "e1", "e2", "e3", "e4",
                          "f12", "f23", "f34"]                         # Names for the fluxes

        self.FluxGroups = {"Ea": [6, 7, 8, 9],                           # Index or indices of fluxes to add to Actual ET
                           "Q": [1, 2, 3, 4, 5]}                        # Index or indices of fluxes to add to Streamflow

    def init(self):
        """
        Initialization function.
        """
        # parameters
        theta = self.theta
        a1, fa, fb, fc, fd, st, f2, f1, f3 = theta[3:]

        # auxiliary parameters
        t2 = f2 * st           # Threshold before surface runoff 2 starts [mm]
        t1 = t2 + f1 * (st - t2)  # Threshold before surface runoff 1 starts [mm]
        t3 = f3 * (st - t1)   # Threshold before intermediate runoff starts [mm]
        t4 = st - t1 - t3     # Threshold before sub-base runoff starts [mm]
        a2 = fa * a1          # Time parameter for surface runoff 2 [d-1]
        b1 = fb * a2          # Time parameter for intermediate runoff 1 [d-1]
        c1 = fc * b1          # Time parameter for sub-base runoff 1 [d-1]
        d1 = fd * c1          # Time parameter for base runoff 1 [d-1]
        self.aux_theta = np.array([t2, t1, t3, t4, a2, b1, c1, d1])

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
        theta = self.theta
        a0, b0, c0, a1 = theta[:4]

        # auxiliary parameters
        aux_theta = self.aux_theta
        t2, t1, t3, t4, a2, b1, c1, d1 = self.aux_theta

        # delta_t
        delta_t = self.delta_t

        # stores
        S1, S2, S3, S4 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        # fluxes functions
        flux_y1 = interflow_8(S1, a1, t1)
        flux_y2 = interflow_8(S1, a2, t2)
        flux_y3 = interflow_8(S2, b1, t3)
        flux_y4 = interflow_8(S3, c1, t4)
        flux_y5 = baseflow_1(d1, S4)
        flux_e1 = evap_1(S1, Ep, delta_t)
        flux_e2 = evap_1(S2, max(0, Ep - flux_e1), delta_t)
        flux_e3 = evap_1(S3, max(0, Ep - flux_e1 - flux_e2), delta_t)
        flux_e4 = evap_1(S4, max(0, Ep - flux_e1 - flux_e2 - flux_e3), delta_t)
        flux_f12 = recharge_3(a0, S1)
        flux_f23 = recharge_3(b0, S2)
        flux_f34 = recharge_3(c0, S3)

        # stores ODEs
        dS1 = P - flux_e1 - flux_f12 - flux_y1 - flux_y2
        dS2 = flux_f12 - flux_e2 - flux_f23 - flux_y3
        dS3 = flux_f23 - flux_e3 - flux_f34 - flux_y4
        dS4 = flux_f34 - flux_e4 - flux_y5

        # outputs
        dS = np.array([dS1, dS2, dS3, dS4])
        fluxes = np.array([flux_y1, flux_y2, flux_y3, flux_y4, flux_y5,
                           flux_e1, flux_e2, flux_e3, flux_e4,
                           flux_f12, flux_f23, flux_f34])

        return dS, fluxes

    def step(self):
        """
        Runs at the end of every timestep.
        """
        pass

