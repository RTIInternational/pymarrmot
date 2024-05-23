import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.capillary import capillary_3
from pymarrmot.models.flux.interflow import interflow_8
from pymarrmot.models.flux.exchange import exchange_2
from pymarrmot.models.flux.evaporation import evap_1
from pymarrmot.models.flux.baseflow import baseflow_1
from pymarrmot.models.flux.recharge import recharge_3

class m_38_tank2_16p_5s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Tank model - SMA

    References:
    - Sugawara, M. (1995). Tank model. In V. P. Singh (Ed.), Computer models of
      watershed hydrology (pp. 165–214). Water Resources Publications, USA.
    """

    def __init__(self):
        """
        creator method
        """
        super().__init__()
        self.num_stores = 5  # number of model stores
        self.num_fluxes = 14  # number of model fluxes
        self.num_params = 16

        self.jacob_pattern = np.array([[1, 0, 0, 0, 1],
                                        [1, 1, 0, 0, 0],
                                        [1, 1, 1, 0, 0],
                                        [1, 1, 1, 1, 0],
                                        [1, 0, 0, 0, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[0, 1],  # a0, Time parameter for drainage 1>2 [d-1]
                                     [0, 1],  # b0, Time parameter for drainage 2>3 [d-1]
                                     [0, 1],  # c0, Time parameter for drainage 3>4 [d-1]
                                     [0, 1],  # a1, Time parameter for surface runoff 1 [d-1]
                                     [0, 1],  # fa, Fraction of a1 that is a2 [-]
                                     [0, 1],  # fb, Fraction of a2 that is b1 [-]
                                     [0, 1],  # fc, Fraction of b1 that is c1 [-]
                                     [0, 1],  # fd, Fraction of c1 that is d1 [-]
                                     [1, 2000],  # st, Maximum soil depth (sum of runoff thresholds) [mm]
                                     [0.01, 0.99],  # f2, Fraction of st that constitutes threshold t2 [-]
                                     [0.01, 0.99],  # f1, Fraction of st-t2 that is added to t2 to find threshold 1 [-] (ensures t1 > t2)
                                     [0.01, 0.99],  # f3, Fraction of st-t1-t2 that consitutes threshold 3 [-]
                                     [0, 4],  # k1, Base rate of capillary rise [mm/d]
                                     [0, 4],  # k2, Base rate of soil moisture exchange [mm/d]
                                     [0.01, 0.99],  # z1, Fraction Stot that is sm1 [-]
                                     [0.01, 0.99]])  # z2, Fraction of Stot-sm1 that is sm2 [-]

        self.store_names = ["S1", "S2", "S3", "S4", "S5"]  # Names for the stores
        self.flux_names = ["t1", "t2",
                           "y1", "y2", "y3", "y4", "y5",
                           "e1", "e2", "e3", "e4",
                           "f12", "f23", "f34"]  # Names for the fluxes

        self.flux_groups = {"Ea": [7, 8, 9, 10],  # Index or indices of fluxes to add to Actual ET
                            "Q": [2, 3, 4, 5, 6]}  # Index or indices of fluxes to add to Streamflow

    def init(self):
        """
        INITialisation function
        """
        # parameters
        theta = self.theta
        a1 = theta[3]  # Time parameter for surface runoff 1 [d-1]
        fa = theta[4]  # Fraction of a1 that is a2 [-]
        fb = theta[5]  # Fraction of a2 that is b1 [-]
        fc = theta[6]  # Fraction of b1 that is c1 [-]
        fd = theta[7]  # Fraction of c1 that is d1 [-]
        st = theta[8]  # Maximum soil depth (sum of runoff thresholds) [mm]
        f2 = theta[9]  # Fraction of st-sm1 that is added to sm1 to find threshold t2 [-] (ensures t2 > sm1)
        f1 = theta[10]  # Fraction of st-t2 that is added to t2 to find threshold 1 [-] (ensures t1 > t2)
        f3 = theta[11]  # Fraction of st-t1-sm2 that consitutes threshold 3 [-]
        z1 = theta[14]  # Fraction st that is sm1 [-]
        z2 = theta[15]  # Fraction of st-t1 that is sm2 [-]

        # auxiliary parameters
        sm1 = z1 * st  # Size of primary soil moisture store, threshold before F12 starts [mm]
        t2 = sm1 + f2 * (st - sm1)  # Threshold before surface runoff Y2 starts [mm]
        t1 = t2 + f1 * (st - t2)  # Thresold before surface runoff Y1 starts [mm]
        sm2 = z2 * (st - t1)  # Size of secondary soil moisture store S5 [mm]
        t3 = f3 * (st - t1 - sm2)  # Threshold before intermediate runoff starts [mm]
        t4 = st - t1 - sm2 - t3  # Threshold before sub-base runoff starts [mm]
        a2 = fa * a1  # Time parameter for surface runoff 2 [d-1]
        b1 = fb * a2  # Time parameter for intermediate runoff 1 [d-1]
        c1 = fc * b1  # Time parameter for sub-base runoff 1 [d-1]
        d1 = fd * c1  # Time parameter for base runoff 1 [d-1]
        self.aux_theta = [sm1, t2, t1, sm2, t3, t4, a2, b1, c1, d1]

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation

        Parameters:
        S (array): state variables

        Returns:
        dS (array): derivatives of state variables
        fluxes (array): model fluxes
        """
        # parameters
        theta = self.theta
        a0 = theta[0]  # Time parameter for drainage 1>2 [d-1]
        b0 = theta[1]  # Time parameter for drainage 2>3 [d-1]
        c0 = theta[2]  # Time parameter for drainage 3>4 [d-1]
        a1 = theta[3]  # Time parameter for surface runoff 1 [d-1]
        k1 = theta[12]  # Base rate of capillary rise [mm/d]
        k2 = theta[13]  # Base rate of soil moisture exchange [mm/d]

        # auxiliary parameters
        aux_theta = self.aux_theta
        sm1 = aux_theta[0]  # Size of primary soil moisture store, threshold before F12 starts [mm]
        t2 = aux_theta[1]  # Threshold before surface runoff Y2 starts [mm]
        t1 = aux_theta[2]  # Thresold before surface runoff Y1 starts [mm]
        sm2 = aux_theta[3]  # Size of secondary soil moisture store S5 [mm]
        t3 = aux_theta[4]  # Threshold before intermediate runoff starts [mm]
        t4 = aux_theta[5]  # Threshold before sub-base runoff starts [mm]
        a2 = aux_theta[6]  # Time parameter for surface runoff 2 [d-1]
        b1 = aux_theta[7]  # Time parameter for intermediate runoff 1 [d-1]
        c1 = aux_theta[8]  # Time parameter for sub-base runoff 1 [d-1]
        d1 = aux_theta[9]  # Time parameter for base runoff 1 [d-1]

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
        flux_t1 = capillary_3(k1, sm1, S1, S2, delta_t)
        flux_t2 = exchange_2(k2, S1, sm1, S5, sm2)
        flux_y1 = interflow_8(S1, a1, t1)
        flux_y2 = interflow_8(S1, a2, t2)
        flux_y3 = interflow_8(S2, b1, t3)
        flux_y4 = interflow_8(S3, c1, t4)
        flux_y5 = baseflow_1(d1, S4)
        flux_e1 = evap_1(S1, Ep, delta_t)
        flux_e2 = evap_1(S2, max(0, Ep - flux_e1), delta_t)
        flux_e3 = evap_1(S3, max(0, Ep - flux_e1 - flux_e2), delta_t)
        flux_e4 = evap_1(S4, max(0, Ep - flux_e1 - flux_e2 - flux_e3), delta_t)
        flux_f12 = interflow_8(S1, a0, sm1)
        flux_f23 = recharge_3(b0, S2)
        flux_f34 = recharge_3(c0, S3)

        # stores ODEs
        dS1 = P + flux_t1 - flux_t2 - flux_e1 - flux_f12 - flux_y1 - flux_y2
        dS2 = flux_f12 - flux_t1 - flux_e2 - flux_f23 - flux_y3
        dS3 = flux_f23 - flux_e3 - flux_f34 - flux_y4
        dS4 = flux_f34 - flux_e4 - flux_y5
        dS5 = flux_t2

        # outputs
        dS = np.array([dS1, dS2, dS3, dS4, dS5])
        fluxes = [flux_t1, flux_t2,
                  flux_y1, flux_y2, flux_y3, flux_y4, flux_y5,
                  flux_e1, flux_e2, flux_e3, flux_e4,
                  flux_f12, flux_f23, flux_f34]

        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep.
        """
        pass
