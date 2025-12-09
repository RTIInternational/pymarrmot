import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.snowfall import snowfall_1
from pymarrmot.models.flux.rainfall import rainfall_1
from pymarrmot.models.flux.split import split_1
from pymarrmot.models.flux.baseflow import baseflow_1
from pymarrmot.models.flux.melt import melt_1
from pymarrmot.models.flux.evaporation import evap_1, evap_15
from pymarrmot.models.flux.interflow import interflow_5, interflow_6
from pymarrmot.models.flux.saturation import saturation_1
from pymarrmot.models.unit_hydro import (uh_6_gamma, update_uh, route)

class m_41_nam_10p_6s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: NAM

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model references:
    Nielsen, S. A., & Hansen, E. (1973). Numerical simulation of the
    rainfall-runoff process on a daily basis. Nordic Hydrology, (4), 171–190.
    http://doi.org/https://doi.org/10.2166/nh.1973.0013
    """

    def __init__(self):
        """
        Creator method
        """
        super().__init__()
        self.num_stores = 6  # number of model stores
        self.num_fluxes = 14  # number of model fluxes
        self.num_params = 10

        self.jacob_pattern = np.array([[1, 0, 0, 0, 0, 0],
                                      [1, 1, 1, 0, 0, 0],
                                      [1, 1, 1, 0, 0, 0],
                                      [1, 1, 1, 1, 0, 0],
                                      [0, 1, 1, 0, 1, 0],
                                      [1, 1, 1, 0, 0, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[0, 20],  # cs, Degree-day factor for snowmelt [mm/oC/d]
                                    [0, 1],   # cif, Runoff coefficient for interflow [d-1]
                                    [1, 2000],  # stot, Maximum total soil moisture depth [mm]
                                    [0, 0.99],  # cl1, Lower zone filling threshold for interflow generation [-]
                                    [0.01, 0.99],  # f1, Fraction of total soil depth that makes up lstar
                                    [0, 1],   # cof, Runoff coefficient for overland flow [d-1]
                                    [0, 0.99],  # cl2, Lower zone filling threshold for overland flow generation [-]
                                    [0, 1],   # k0, Overland flow routing delay [d-1]
                                    [0, 1],   # k1, Interflow routing delay [d-1]
                                    [0, 1]])  # kb, Baseflow routing delay [d-1]

        self.store_names = ["S1", "S2", "S3", "S4", "S5", "S6"]  # Names for the stores
        self.flux_names = ["ps", "pr", "m", "eu", "pn", "of", "inf",
                          "if", "dl", "gw", "el", "qo", "qi", "qb"]  # Names for the fluxes

        self.flux_groups = {"Ea": [4, 11],  # Index or indices of fluxes to add to Actual ET
                           "Q": [12, 13, 14]}  # Index or indices of fluxes to add to Streamflow

        self.aux_theta = None  # Auxiliary parameters

    def init(self):
        """
        INIT is run automatically as soon as both theta and delta_t are
        set (it is therefore ran only once at the beginning of the run.
        Use it to initialise all the model parameters (in case there are
        derived parameters) and unit hydrographs and set minima and
        maxima for stores based on parameters.
        """
        theta = self.theta
        stot = theta[2]  # Maximum total soil moisture depth [mm]
        fl = theta[4]  # Fraction of total soil depth that makes up lstar

        # set auxiliary parameters
        lstar = fl * stot  # Maximum lower zone storage [mm]
        ustar = (1 - fl) * stot  # Upper zone maximum storage [mm]
        self.aux_theta = [lstar, ustar]

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation
        """
        theta = self.theta
        cs = theta[0]  # Degree-day factor for snowmelt [mm/oC/d]
        cif = theta[1]  # Runoff coefficient for interflow [d-1]
        cl1 = theta[3]  # Lower zone filling threshold for interflow generation [-]
        cof = theta[5]  # Runoff coefficient for overland flow [d-1]
        cl2 = theta[6]  # Lower zone filling threshold for overland flow generation [-]
        k0 = theta[7]  # Overland flow routing delay [d-1]
        k1 = theta[8]  # Interflow routing delay [d-1]
        kb = theta[9]  # Baseflow routing delay [d-1]

        # auxiliary parameters
        aux_theta = self.aux_theta
        lstar = aux_theta[0]  # Maximum lower zone storage [mm]
        ustar = aux_theta[1]  # Upper zone maximum storage [mm]

        # delta_t
        delta_t = self.delta_t

        # stores
        S1, S2, S3, S4, S5, S6 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        # fluxes functions
        flux_ps = snowfall_1(P, T, 0)
        flux_pr = rainfall_1(P, T, 0)
        flux_m = melt_1(cs, 0, T, S1, delta_t)
        flux_eu = evap_1(S2, Ep, delta_t)
        flux_pn = saturation_1(flux_pr + flux_m, S2, ustar)
        flux_of = interflow_6(cof, cl2, flux_pn, S3, lstar)
        flux_inf = flux_pn - flux_of
        flux_if = interflow_6(cif, cl1, S2, S3, lstar)
        flux_dl = split_1(1 - S3 / lstar, flux_inf)
        flux_gw = split_1(S3 / lstar, flux_inf)
        flux_el = evap_15(Ep, S3, lstar, S2, 0.01, delta_t)
        flux_qo = interflow_5(k0, S4)
        flux_qi = interflow_5(k1, S5)
        flux_qb = baseflow_1(kb, S6)

        # stores ODEs
        dS1 = flux_ps - flux_m
        dS2 = flux_pr + flux_m - flux_eu - flux_if - flux_pn
        dS3 = flux_dl - flux_el
        dS4 = flux_of - flux_qo
        dS5 = flux_if - flux_qi
        dS6 = flux_gw - flux_qb

        # outputs
        dS = np.array([dS1, dS2, dS3, dS4, dS5, dS6])
        fluxes = np.array([flux_ps, flux_pr, flux_m, flux_eu, flux_pn,
                           flux_of, flux_inf, flux_if, flux_dl, flux_gw,
                           flux_el, flux_qo, flux_qi, flux_qb])
        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep.
        """
        pass

