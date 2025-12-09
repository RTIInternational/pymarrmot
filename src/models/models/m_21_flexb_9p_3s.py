import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.saturation import saturation_3
from pymarrmot.models.flux.evaporation import evap_3
from pymarrmot.models.flux.percolation import percolation_2
from pymarrmot.models.flux.split import split_1
from pymarrmot.models.flux.baseflow import baseflow_1
from pymarrmot.models.unit_hydro import route, uh_3_half, update_uh

class m_21_flexb_9p_3s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Flex-B

    Model reference:
    Fenicia, F., McDonnell, J. J., & Savenije, H. H. G. (2008). Learning from
    model improvement: On the contribution of complementary data to process
    understanding. Water Resources Research, 44(6), 1–13.
    http://doi.org/10.1029/2007WR006386
    """

    def __init__(self):
        """
        Creator method
        """
        super().__init__()
        self.num_stores = 3  # number of model stores
        self.num_fluxes = 9  # number of model fluxes
        self.num_params = 9  # number of model parameters

        self.jacob_pattern = np.array([[1, 0, 0],
                                      [1, 1, 0],
                                      [1, 0, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[1, 2000],    # URmax, Maximum soil moisture storage [mm]
                                    [0, 10],       # beta, Unsaturated zone shape parameter [-]
                                    [0, 1],        # D, Fast/slow runoff distribution parameter [-]
                                    [0, 20],       # PERCmax, Maximum percolation rate [mm/d]
                                    [0.05, 0.95],  # Lp, Wilting point as fraction of s1max [-]
                                    [1, 5],        # Nlagf, Flow delay before fast runoff [d]
                                    [1, 15],       # Nlags, Flow delay before slow runoff [d]
                                    [0, 1],        # Kf, Fast runoff coefficient [d-1]
                                    [0, 1]])       # Ks, Slow runoff coefficient [d-1]

        self.store_names = ["S1", "S2", "S3"]  # Names for the stores
        self.flux_names = ["ru", "eur", "ps", "rf", "rs",
                          "rfl", "rsl", "qf", "qs"]  # Names for the fluxes

        self.flux_groups = {"Ea": 2,      # Index or indices of fluxes to add to Actual ET
                           "Q": [8, 9]}  # Index or indices of fluxes to add to Streamflow

    def init(self):
        """
        INITialisation function
        """
        # parameters
        theta = self.theta
        delta_t = self.delta_t
        nlagf = theta[5]  # Flow delay before fast runoff [d]
        nlags = theta[6]  # Flow delay before slow runoff [d]

        # initialise the unit hydrographs and still-to-flow vectors
        uh_f = uh_3_half(nlagf, delta_t)
        uh_s = uh_3_half(nlags, delta_t)

        self.uhs = (uh_f, uh_s)

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation

        Parameters:
        S : array-like
            State variables

        Returns:
        tuple: Tuple containing two arrays (dS, fluxes)
               dS: Array of derivatives of state variables
               fluxes: Array of fluxes
        """
        # parameters
        theta = self.theta
        s1max = theta[0]
        beta = theta[1]
        d = theta[2]
        percmax = theta[3]
        lp = theta[4]
        kf = theta[7]
        ks = theta[8]

        # delta_t
        delta_t = self.delta_t

        # unit hydrographs and still-to-flow vectors
        uh_f, uh_s = self.uhs

        # stores
        S1, S2, S3 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        # fluxes functions
        flux_ru = saturation_3(S1, s1max, beta, P)
        flux_eur = evap_3(lp, S1, s1max, Ep, delta_t)
        flux_ps = percolation_2(percmax, S1, s1max, delta_t)
        flux_rf = split_1(1 - d, P - flux_ru)
        flux_rs = split_1(d, P - flux_ru)
        flux_rfl = route(flux_rf, uh_f)
        flux_rsl = route(flux_ps + flux_rs, uh_s)
        flux_qf = baseflow_1(kf, S2)
        flux_qs = baseflow_1(ks, S3)
        
        # stores ODEs
        dS1 = flux_ru - flux_eur - flux_ps
        dS2 = flux_rfl - flux_qf
        dS3 = flux_rsl - flux_qs

        # outputs
        dS = np.array([dS1, dS2, dS3])
        fluxes = np.array([flux_ru, flux_eur, flux_ps, flux_rf, flux_rs,
                           flux_rfl, flux_rsl, flux_qf, flux_qs])

        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep.
        """
        # unit hydrographs and still-to-flow vectors
        uh_f, uh_s = self.uhs

        # input fluxes to the unit hydrographs
        fluxes = self.fluxes[self.t]
        flux_ps = fluxes[2]
        flux_rf = fluxes[3]
        flux_rs = fluxes[4]

        # update still-to-flow vectors using fluxes at current step and unit hydrographs
        uh_f = update_uh(uh_f, flux_rf)
        uh_s = update_uh(uh_s, flux_ps + flux_rs)

        self.uhs = (uh_f, uh_s)

       
