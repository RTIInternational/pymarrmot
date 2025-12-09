import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.baseflow import baseflow_7
from pymarrmot.models.flux.interflow import interflow_3
from pymarrmot.models.flux.split import split_1
from pymarrmot.models.flux.effective_1 import effective_1
from pymarrmot.models.flux.evaporation import evap_1, evap_3
from pymarrmot.models.flux.interception import interception_1
from pymarrmot.models.flux.saturation import saturation_13
from pymarrmot.models.flux.recharge import recharge_3

class m_42_hycymodel_12p_6s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: HYCYMODEL

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model reference:
    Fukushima, Y. (1988). A model of river flow forecasting for a small
    forested mountain catchment. Hydrological Processes, 2(2), 167–185.
    """

    def __init__(self):
        """
        Creator method.
        """
        super().__init__()
        self.num_stores = 6  # number of model stores
        self.num_fluxes = 18  # number of model fluxes
        self.num_params = 12

        self.jacob_pattern = [[1, 0, 0, 0, 0, 0],
                             [1, 1, 0, 0, 0, 0],
                             [1, 1, 1, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0],
                             [1, 1, 1, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1]]  # Jacobian matrix of model store ODEs

        self.par_ranges = [[0, 1],  # c,    Fraction area that is channel [-]
                          [0, 5],  # imax, Maximum total interception storage [mm]
                          [0, 1],  # a,    Fraction stem/trunk interception [-]
                          [0.01, 0.99],  # fi2,  Fraction of total interception that is trunk/stem interception [mm]
                          [0, 1],  # kin,  Infiltration runoff coefficient [d-1]
                          [1, 2000],  # D50,  Soil depth where 50% of area contributes to effective flow [mm]
                          [0.01, 0.99],  # fd16, Soil depth where 16% of area contributes to effective flow [mm]
                          [1, 2000],  # sbc,  Soil depth where evaporation rate starts to decline [mm]
                          [0, 1],  # kb,   Baseflow runoff coefficient [d-1]
                          [1, 5],  # pb,   Baseflow non-linearity [-]
                          [0, 1],  # kh,   Hillslope runoff coefficient [d-1]
                          [0, 1]]  # kc,   Channel runoff coefficient [d-1]

        self.store_names = ["S1", "S2", "S3", "S4", "S5", "S6"]  # Names for the stores
        self.flux_names = ["rc", "rg", "eic", "qie", "qis", "rt",
                          "eis", "rs", "rn", "esu", "re", "qin",
                          "esb", "qb", "qh", "qc", "ec", "qt"]  # Names for the fluxes

        self.flux_groups = {'Ea': [3, 7, 10, 13, 17],  # Index or indices of fluxes to add to Actual ET
                           'Q': [18]}  # Index or indices of fluxes to add to Streamflow

    def init(self):
        """
        Initialization function.
        """
        theta = self.theta
        imax = theta[1]  # Maximum total interception storage [mm]
        fi2 = theta[3]  # Fraction of total interception that is trunk/stem interception [mm]
        d50 = theta[5]  # Soil depth where 50% of area contributes to effective flow [mm]
        fd16 = theta[6]  # Fraction of D50 that is D16 [-]

        # Auxiliary parameters
        i1max = (1 - fi2) * imax  # Maximum canopy interception [mm]
        i2max = fi2 * imax  # Maximum trunk/stem interception [mm]
        d16 = fd16 * d50  # Soil depth where 16% of area contributes to effective flow [mm]
        self.aux_theta = [i1max, i2max, d16]

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation.

        Parameters:
        S : array_like
            State variables.

        Returns:
        dS : array_like
            Derivative of state variables.
        fluxes : array_like
            Fluxes.
        """
        theta = self.theta
        c = theta[0]  # Fraction area that is channel [-]
        imax = theta[1]  # Maximum total interception storage [mm]
        a = theta[2]  # Fraction stem/trunk interception [-]
        fi2 = theta[3]  # Fraction of total interception that is trunk/stem interception [mm]
        kin = theta[4]  # Infiltration runoff coefficient [d-1]
        d50 = theta[5]  # Soil depth where 50% of area contributes to effective flow [mm]
        fd16 = theta[6]  # Fraction of D50 that is D16 [-]
        sbc = theta[7]  # Soil depth where evaporation rate starts to decline [mm]
        kb = theta[8]  # Baseflow runoff coefficient [d-1]
        pb = theta[9]  # Baseflow non-linearity [-]
        kh = theta[10]  # Hillslope runoff coefficient [d-1]
        kc = theta[11]  # Channel runoff coefficient [d-1]

        # Auxiliary parameters
        aux_theta = self.aux_theta
        i1max = aux_theta[0]  # Maximum canopy interception [mm]
        i2max = aux_theta[1]  # Maximum trunk/stem interception [mm]
        d16 = aux_theta[2]  # Soil depth where 16% of area contributes to effective flow [mm]

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
        flux_rc = split_1(c, P)
        flux_rg = split_1(1 - c, P)
        flux_eic = evap_1(S1, (1 - c) * Ep, delta_t)
        flux_qie = interception_1(flux_rg, S1, i1max)
        flux_qis = split_1(a, flux_qie)
        flux_rt = split_1(1 - a, flux_qie)
        flux_eis = evap_1(S2, (1 - c) * Ep, delta_t)
        flux_rs = interception_1(flux_qis, S2, i2max)
        flux_rn = flux_rt + flux_rs
        flux_esu = evap_1(S3, (1 - c) * Ep, delta_t)
        flux_re = saturation_13(d50, d16, S3, flux_rn)
        flux_qin = recharge_3(kin, S3)
        flux_esb = evap_3(1, S4, sbc, max(0, (1 - c) * Ep - flux_esu), delta_t)
        flux_qb = baseflow_7(kb, pb, S4, delta_t)
        flux_qh = interflow_3(kh, 5 / 3, S5, delta_t)
        flux_qc = interflow_3(kc, 5 / 3, S6, delta_t)
        flux_qt = effective_1(flux_qb + flux_qh + flux_qc, c * Ep)
        flux_ec = (flux_qb + flux_qh + flux_qc) - flux_qt

        # stores ODEs
        dS1 = flux_rg - flux_eic - flux_qie
        dS2 = flux_qis - flux_eis - flux_rs
        dS3 = flux_rn - flux_re - flux_esu - flux_qin
        dS4 = flux_qin - flux_esb - flux_qb
        dS5 = flux_re - flux_qh
        dS6 = flux_rc - flux_qc

        # outputs
        dS = np.array([dS1, dS2, dS3, dS4, dS5, dS6])
        fluxes = [flux_rc, flux_rg, flux_eic, flux_qie, flux_qis, flux_rt,
                  flux_eis, flux_rs, flux_rn, flux_esu, flux_re, flux_qin,
                  flux_esb, flux_qb, flux_qh, flux_qc, flux_ec, flux_qt]
        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep.
        """
        pass
