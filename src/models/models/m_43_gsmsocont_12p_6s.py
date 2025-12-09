import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.snowfall import snowfall_1
from pymarrmot.models.flux.interflow import interflow_3
from pymarrmot.models.flux.split import split_1
from pymarrmot.models.flux.effective_1 import effective_1
from pymarrmot.models.flux.rainfall import rainfall_1
from pymarrmot.models.flux.melt import melt_1, melt_3
from pymarrmot.models.flux.saturation import saturation_9
from pymarrmot.models.flux.baseflow import baseflow_1
from pymarrmot.models.flux.infiltration import infiltration_6
from pymarrmot.models.flux.evaporation import evap_19

class m_43_gsmsocont_12p_6s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: GSM-SOCONT

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model reference:
    Schaefli, B., Hingray, B., Niggli, M., & Musy, a. (2005). A conceptual
    glacio-hydrological model for high mountainous catchments. Hydrology and
    Earth System Sciences Discussions, 2(1), 73–117.
    http://doi.org/10.5194/hessd-2-73-2005
    """

    def __init__(self):
        super().__init__()
        self.num_stores = 6  # number of model stores
        self.num_fluxes = 19  # number of model fluxes
        self.num_params = 12

        self.jacob_pattern = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1]
        ]  # Jacobian matrix of model store ODEs

        self.par_ranges = [
            [0, 1],     # fice,   Fraction of catchment covered by glacier [-]
            [-3, 5],    # t0,     Threshold temperature for snowfall [oC]
            [0, 20],    # asnow,  Degree-day factor for snow melt [mm/oC/d]
            [-3, 3],    # tm,     Threshold temperature for snow melt [oC]
            [0, 1],     # ks,     Runoff coeficient for snow melt on glacier [d-1]
            [0, 20],    # aice,   Degree-day factor for ice melt [mm/oC/d]
            [0, 1],     # ki,     Runoff coeficient for ice melt on glacier [d-1]
            [1, 2000],  # a,      Maximum soil moisture storage [mm]
            [0, 10],    # x,      Evaporation non-linearity [-]
            [0, 5],     # y,      Infiltration non-linearity [-]
            [0, 1],     # ksl,    Runoff coefficient for baseflow [d-1]
            [0, 1]      # beta,   Runoff coefficient for quick flow [mm^(4/3)/d]
        ]

        self.store_names = ["S1", "S2", "S3", "S4", "S5", "S6"]  # Names for the stores
        self.flux_names = [
            "pice", "pices", "picer", "mis", "pirs",
            "piri", "mii", "qis", "qii", "pni",
            "pnis", "pnir", "mnis", "peq", "peff",
            "pinf", "et", "qsl", "qqu"
        ]  # Names for the fluxes

        self.flux_groups = {
            "Ea": [17],                             # Index or indices of fluxes to add to Actual ET
            "Q": [8, 9, 18, 19],                    # Index or indices of fluxes to add to Streamflow
            "GlacierMelt": -7                       # Index of flows from glacier melt
        }

    def init(self):
        """
        Initialization function
        """
        pass

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation
        """
        # parameters
        theta = self.theta
        fice = theta[0]     # Fraction of catchment covered by glacier [-]
        t0 = theta[1]     # Threshold temperature for snowfall [oC]
        asnow = theta[2]     # Degree-day factor for snow melt [mm/oC/d]
        tm = theta[3]     # Threshold temperature for snow melt [oC]
        ks = theta[4]     # Runoff coeficient for snow melt on glacier [d-1]
        aice = theta[5]     # Threshold temperature for ice melt [oC]
        ki = theta[6]     # Runoff coeficient for ice melt on glacier [d-1]
        a = theta[7]     # Maximum soil moisture storage [mm]
        x = theta[8]     # Evaporation non-linearity [-]
        y = theta[9]    # Infiltration non-linearity [-]
        ksl = theta[10]    # Runoff coefficient for baseflow [d-1]
        beta = theta[11]    # Runoff coefficient for quick flow [mm^(4/3)/d]
        # delta_t
        delta_t = self.delta_t
        # stores
        S1 = S[0]
        S2 = S[1]
        S3 = S[2]
        S4 = S[3]
        S5 = S[4]
        S6 = S[5]
        
        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]
        
        # fluxes functions
        flux_pice = split_1(fice, P)
        flux_pices = snowfall_1(flux_pice, T, t0)
        flux_picer = rainfall_1(flux_pice, T, t0)
        flux_mis = melt_1(asnow, tm, T, S1, delta_t)
        flux_pirs = saturation_9(flux_picer, S1, 0.01)
        flux_piri = effective_1(flux_picer, flux_pirs)
        #flux_mii = melt_3(aice, tm, T, Inf, S1, 0.01, delta_t)
        #original code, which I believe is incorrect (SAS)
        flux_mii = melt_3(aice, tm, T, np.inf, S1, 0.01, delta_t)
        flux_qis = baseflow_1(ks, S2)
        flux_qii = baseflow_1(ki, S3)
        flux_pni = split_1(1 - fice, P)
        flux_pnis = snowfall_1(flux_pni, T, t0)
        flux_pnir = rainfall_1(flux_pni, T, t0)
        flux_mnis = melt_1(asnow, tm, T, S4, delta_t)
        flux_peq = flux_pnir + flux_mnis
        flux_peff = infiltration_6(1, y, S5, a, flux_peq)
        flux_pinf = effective_1(flux_peq, flux_peff)
        flux_et = evap_19(1, x, S5, a, Ep, delta_t)
        flux_qsl = baseflow_1(ksl, S5)
        flux_qqu = interflow_3(beta, 5 / 3, S6, delta_t)
        # stores ODEs
        dS1 = flux_pices - flux_mis
        dS2 = flux_pirs + flux_mis - flux_qis
        dS3 = flux_piri + flux_mii - flux_qii
        dS4 = flux_pnis - flux_mnis
        dS5 = flux_pinf - flux_et - flux_qsl
        dS6 = flux_peff - flux_qqu
        # outputs
        dS = np.array([dS1, dS2, dS3, dS4, dS5, dS6])
        fluxes = np.array([flux_pice, flux_pices, flux_picer, flux_mis, flux_pirs,
                           flux_piri, flux_mii, flux_qis, flux_qii, flux_pni,
                           flux_pnis, flux_pnir, flux_mnis, flux_peq, flux_peff,
                           flux_pinf, flux_et, flux_qsl, flux_qqu])
        return dS, fluxes

    def step(self):
        """
        Function runs at the end of every timestep
        """
        pass
