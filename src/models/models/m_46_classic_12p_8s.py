import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation import evap_18, evap_1
from pymarrmot.models.flux.split import split_1
from pymarrmot.models.flux.effective_1 import effective_1
from pymarrmot.models.flux.saturation import saturation_1, saturation_9
from pymarrmot.models.flux.baseflow import baseflow_1

class m_46_classic_12p_8s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: CLASSIC

    Model reference:
    Crooks, S. M., & Naden, P. S. (2007). CLASSIC: a semi-distributed
    rainfall-runoff modelling system. Hydrology and Earth System Sciences,
    11(1), 516–531. http://doi.org/10.5194/hess-11-516-2007
    """
    def __init__(self):
        """
        Constructor for m_46_classic_12p_8s.

        Initializes model-specific attributes.
        """
        super().__init__()
        self.aux_theta = None  # Auxiliary parameters

        # Model attributes
        self.num_stores = 8  # number of model stores
        self.num_fluxes = 21  # number of model fluxes
        self.num_params = 12

        self.jacob_pattern = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0, 0, 0],
                                       [0, 0, 0, 1, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 1, 1, 0, 0],
                                       [0, 0, 0, 1, 1, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 1]])
        # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[0, 1], [0.01, 0.99], [1, 2000], [0, 1],
                                    [0, 1], [0, 1], [0.01, 0.99], [1, 2000],
                                    [0, 1], [0, 1], [0, 1], [0, 1]])
        # Parameter ranges

        self.store_names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
        # Names for the stores

        self.flux_names = ["pp", "ps", "pi", "epx", "ppx", "epy", "ppe", "q",
                          "psd", "psi", "esx", "psx", "esy", "pse", "psq",
                          "pss", "xq", "xs", "ei", "pie", "u"]
        # Names for the fluxes

        self.flux_groups = {"Ea": [4, 6, 11, 13, 19],
                           "Q": [8, 17, 18, 21]}
        # Index or indices of fluxes for specific groups

        self.StoreSigns = [1, -1, 1, 1, -1, 1, 1, 1]
        # Signs to give to stores (-1 is a deficit store)

    def init(self):
        """
        Initialization function.

        Initializes parameters and auxiliary parameters.
        """
        theta = self.theta
        fap = theta[0]  # Fraction of catchment area that has permeable soils [-]
        fdp = theta[1]  # Fraction of depth of permeable soil that is store Px [-]
        dp = theta[2]   # Depth of permeable soil [mm]
        tf = theta[5]   # Fraction of (1-fap) that is fas [-]
        fds = theta[6]  # Fraction of depth of semi-permeable soil that is store Sx [-]
        ds = theta[7]   # Depth of semi-permeable soil [mm]

        # auxiliary parameters;
        fas = (1 - fap) * tf  # Fraction of catchment area that has semi-permeable soils [-]
        fai = 1 - fap - fas   # Fraction of catchment area that has impermeable soils [-]
        pxm = fdp * dp        # Depth of store Px [mm]
        pym = (1 - fdp) * dp  # Depth of store Py [mm]
        sxm = fds * ds        # Depth of store Sx [mm]
        sym = (1 - fds) * ds  # Depth of store Sy [mm]
        self.aux_theta = [fas, fai, pxm, pym, sxm, sym]

    def model_fun(self, S):
        """
        Model governing equations in state-space formulation.

        :param S: State variables
        :return: Time derivatives of state variables, Fluxes
        """
        theta = self.theta
        fap = theta[0]    # Fraction of catchment area that has permeable soils [-]
        fdp = theta[1]    # Fraction of depth of permeable soil that is store Px [-]
        dp = theta[2]     # Depth of permeable soil [mm]
        cq = theta[3]     # Runoff coefficient for permeable soil [d-1]
        d1 = theta[4]     # Fraction of Ps that infiltrates into semi-permeable soil [-]
        tf = theta[5]     # Fraction of (1-fap) that is fas [-]
        fds = theta[6]    # Fraction of depth of semi-permeable soil that is store Sx [-]
        ds = theta[7]     # Depth of semi-permeable soil [mm]
        d2 = theta[8]     # Fraction effective precipitation in semi-permeable soils that goes to quick flow [-]
        cxq = theta[9]    # Quick runoff coefficient for semi-permeable soil [d-1]
        cxs = theta[10]   # Slow runoff coefficient for semi-permeable soil [d-1]
        cu = theta[11]    # Runoff coefficient for impermeable soil [d-1]

        aux_theta = self.aux_theta
        fas = aux_theta[0]   # Fraction of catchment area that has semi-permeable soils [-]
        fai = aux_theta[1]   # Fraction of catchment area that has impermeable soils [-]
        pxm = aux_theta[2]   # Depth of store Px [mm]
        pym = aux_theta[3]   # Depth of store Py [mm]
        sxm = aux_theta[4]   # Depth of store Sx [mm]
        sym = aux_theta[5]   # Depth of store Sy [mm]

        delta_t = self.delta_t  # delta_t

        S1 = S[0]
        S2 = S[1]
        S3 = S[2]
        S4 = S[3]
        S5 = S[4]
        S6 = S[5]
        S7 = S[6]
        S8 = S[7]

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        # Fluxes functions
        flux_pp = split_1(fap, P)
        flux_ps = split_1(fas, P)
        flux_pi = split_1(fai, P)
        flux_epx = evap_1(S1, fap * Ep, delta_t)
        flux_ppx = saturation_1(flux_pp, S1, pxm)
        flux_epy = evap_18(1.9, 0.6523, pxm, S2 + pxm, fap * Ep - flux_epx)
        flux_ppe = saturation_9(flux_ppx, S2, 0.01)
        flux_q = baseflow_1(cq, S3)
        flux_psd = split_1(1 - d1, flux_ps)
        flux_psi = split_1(d1, flux_ps)
        flux_esx = evap_1(S4, fas * Ep, delta_t)
        flux_psx = saturation_1(flux_psi, S4, sxm)
        flux_esy = evap_18(1.9, 0.6523, sxm, S4 + sxm, fas * Ep - flux_esx)
        flux_pse = saturation_9(flux_psx, S5, 0.01)
        flux_psq = split_1(d2, flux_pse + flux_psd)
        flux_pss = split_1(1 - d2, flux_pse + flux_psd)
        flux_xq = baseflow_1(cxq, S6)
        flux_xs = baseflow_1(cxs, S7)
        flux_pie = effective_1(fai * flux_pi, 0.5)
        flux_ei = flux_pi - flux_pie
        flux_u = baseflow_1(cu, S8)

        # Stores ODEs
        dS1 = flux_pp - flux_epx - flux_ppx
        dS2 = -(flux_ppx - flux_epy - flux_ppe)
        dS3 = flux_ppe - flux_q
        dS4 = flux_psi - flux_esx - flux_psx
        dS5 = -(flux_psx - flux_esy - flux_pse)
        dS6 = flux_psq - flux_xq
        dS7 = flux_pss - flux_xs
        dS8 = flux_pie - flux_u

        # Outputs
        dS = np.array([dS1, dS2, dS3, dS4, dS5, dS6, dS7, dS8])
        fluxes = [flux_pp, flux_ps, flux_pi, flux_epx, flux_ppx,
                  flux_epy, flux_ppe, flux_q, flux_psd, flux_psi,
                  flux_esx, flux_psx, flux_esy, flux_pse, flux_psq,
                  flux_pss, flux_xq, flux_xs, flux_ei, flux_pie, flux_u]

        return dS, fluxes

    def step(self):
        """
        Function that runs at the end of every timestep.
        """
        pass
