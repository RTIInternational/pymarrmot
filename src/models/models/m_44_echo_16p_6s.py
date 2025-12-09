import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.snowfall import snowfall_1
from pymarrmot.models.flux.rainfall import rainfall_1
from pymarrmot.models.flux.melt import melt_1, melt_2
from pymarrmot.models.flux.effective_1 import effective_1
from pymarrmot.models.flux.evaporation import evap_1, evap_22
from pymarrmot.models.flux.interception import interception_1
from pymarrmot.models.flux.refreeze_1 import refreeze_1
from pymarrmot.models.flux.saturation import saturation_1
from pymarrmot.models.flux.excess_1 import excess_1
from pymarrmot.models.flux.infiltration import infiltration_4
from pymarrmot.models.flux.recharge import recharge_6, recharge_7
from pymarrmot.models.flux.baseflow import baseflow_1

class m_44_echo_16p_6s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: ECHO

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model references
    Schaefli, B., Hingray, B., Niggli, M., & Musy, a. (2005). A conceptual 
    glacio-hydrological model for high mountainous catchments. Hydrology and 
    Earth System Sciences Discussions, 2(1), 73–117. 
    http://doi.org/10.5194/hessd-2-73-2005
    
    Schaefli, B., Nicotina, L., Imfeld, C., Da Ronco, P., Bertuzzo, E., & 
    Rinaldo, A. (2014). SEHR-ECHO v1.0: A spatially explicit hydrologic 
    response model for ecohydrologic applications. Geoscientific Model 
    Development, 7(6), 2733–2746. http://doi.org/10.5194/gmd-7-2733-2014
    """
    def __init__(self):
        """
        creator method
        """
        super().__init__()
        self.aux_theta = None

        self.num_stores = 6
        self.num_fluxes = 20
        self.num_params = 16
        self.jacob_pattern = [[1,0,0,0,0,0],
                             [1,1,1,0,0,0],
                             [1,1,1,0,0,0],
                             [1,1,1,1,0,0],
                             [0,0,0,1,1,0],
                             [0,0,0,1,0,1]]
        self.par_ranges = [[0, 5],
                          [-3, 5],
                          [-3, 3],
                          [0, 20],
                          [0, 1],
                          [0, 2],
                          [0, 1],
                          [0, 200],
                          [1, 2000],
                          [0.05, 0.95],
                          [0.05, 0.95],
                          [0, 1],
                          [0, 5],
                          [0, 20],
                          [0, 1],
                          [0, 1]]
        self.store_names = ["S1", "S2", "S3", "S4", "S5", "S6"]
        self.flux_names = ["ei", "pn", "ps", "pr", "ms",
                          "fs", "gs", "mw", "ew", "eq",
                          "rh", "eps", "et", "fi", "rd",
                          "l", "lf", "ls", "rf", "rs"]
        self.flux_groups = {"Ea": [1, 13],
                           "Q": [11, 15, 19, 20]}

    def init(self):
        """
        INITialisation function
        """
        theta = self.theta
        smax = theta[8]
        fsm = theta[9]
        fsw = theta[10]
        sm = fsm * smax
        sw = fsw * sm
        self.aux_theta = [sm, sw]

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation
        """
        theta = self.theta
        rho = theta[0]
        ts = theta[1]
        tm = theta[2]
        as_ = theta[3]
        af = theta[4]
        gmax = theta[5]
        the = theta[6]
        phi = theta[7]
        smax = theta[8]
        fsm = theta[9]
        fsw = theta[10]
        ksat = theta[11]
        c = theta[12]
        lmax = theta[13]
        kf = theta[14]
        ks = theta[15]
        aux_theta = self.aux_theta
        sm = aux_theta[0]
        sw = aux_theta[1]
        delta_t = self.delta_t
        S1, S2, S3, S4, S5, S6 = S
        t = self.t
        if t == 1:
            S3old = self.S0[2]
        else:
            S3old = self.stores[t-1][2]
       
        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        flux_ei = evap_1(S1, Ep, delta_t)
        flux_pn = interception_1(P, S1, rho)
        flux_ps = snowfall_1(flux_pn, T, ts)
        flux_pr = rainfall_1(flux_pn, T, ts)
        flux_ms = melt_1(as_, tm, T, S2, delta_t)
        flux_fs = refreeze_1(af, as_, tm, T, S3, delta_t)
        flux_gs = melt_2(gmax, S2, delta_t)
        flux_mw = saturation_1(flux_pr + flux_ms, S3, the * S2)
        flux_ew = excess_1(S3old, the * S2, delta_t)
        flux_eq = flux_mw + flux_gs + flux_ew
        flux_fi = infiltration_4(flux_eq, phi)
        flux_rh = effective_1(flux_eq, flux_fi)
        flux_eps = effective_1(Ep, flux_ei)
        flux_et = evap_22(sw, sm, S4, flux_eps, delta_t)
        flux_rd = saturation_1(flux_fi, S4, smax)
        flux_l = recharge_6(ksat, c, S4, delta_t)
        flux_ls = recharge_7(lmax, flux_l)
        flux_lf = effective_1(flux_l, flux_ls)
        flux_rf = baseflow_1(kf, S5)
        flux_rs = baseflow_1(ks, S6)
        dS1 = P - flux_ei - flux_pn
        dS2 = flux_ps + flux_fs - flux_ms - flux_gs
        dS3 = flux_pr + flux_ms - flux_fs - flux_mw - flux_ew
        dS4 = flux_fi - flux_et - flux_rd - flux_l
        dS5 = flux_lf - flux_rf
        dS6 = flux_ls - flux_rs
        dS = np.array([dS1, dS2, dS3, dS4, dS5, dS6])
        fluxes = [flux_ei, flux_pn, flux_ps, flux_pr, flux_ms,
                  flux_fs, flux_gs, flux_mw, flux_ew, flux_eq,
                  flux_rh, flux_eps, flux_et, flux_fi, flux_rd,
                  flux_l, flux_lf, flux_ls, flux_rf, flux_rs]
        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep.
        """
        pass


