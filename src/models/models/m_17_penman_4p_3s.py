import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation import evap_1, evap_16
from pymarrmot.models.flux.saturation import saturation_1, saturation_9
from pymarrmot.models.flux.baseflow import baseflow_1
from pymarrmot.models.flux.split import split_1

class m_17_penman_4p_3s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Penman

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model reference
    Penman, H. L. (1950). the Dependence of Transpiration on Weather and Soil
    Conditions. Journal of Soil Science, 1(1), 74–89.
    http://doi.org/10.1111/j.1365-2389.1950.tb00720.x

    Wagener, T., Lees, M. J., & Wheater, H. S. (2002). A toolkit for the
    development and application of parsimonious hydrological models. In Singh,
    Frevert, & Meyer (Eds.), Mathematical Models of Small Watershed Hydrology
    - Volume 2 (pp. 91–139). Water Resources Publications LLC, USA.
    """

    def __init__(self):
        """
        creator method
        """
        super().__init__()
        self.num_stores = 3  # number of model stores
        self.num_fluxes = 7  # number of model fluxes
        self.num_params = 4

        self.jacob_pattern = np.array([[1, 0, 0],
                                      [1, 1, 0],
                                      [1, 1, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[1, 2000],  # smax, Maximum soil moisture storage [mm]
                                    [0, 1],      # phi, Fraction of direct runoff [-]
                                    [0, 1],      # gam, Evaporation reduction in lower zone [-]
                                    [0, 1]])     # k1, Runoff coefficient [d-1]

        self.store_names = ["S1", "S2", "S3"]  # Names for the stores
        self.flux_names = ["ea", "qex", "u1", "q12", "et", "u2", "q"]  # Names for the fluxes

        self.flux_groups = {"Ea": [1, 5],  # Index or indices of fluxes to add to Actual ET
                           "Q": [7]}      # Index or indices of fluxes to add to Streamflow

        self.StoreSigns = [1, -1, 1]  # Signs to give to stores (-1 is a deficit store), only needed for water balance
    
    def init(self):
        pass
    
    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation
        """
        # parameters
        theta = self.theta
        smax = theta[0]  # Maximum soil moisture storage [mm]
        phi = theta[1]   # Fraction of direct runoff [-]
        gam = theta[2]   # Evaporation reduction in lower zone [-]
        k1 = theta[3]    # Runoff coefficient [d-1]

        # delta_t
        delta_t = self.delta_t

        # stores
        S1, S2, S3 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        # fluxes functions
        flux_ea = evap_1(S1, Ep, delta_t)
        flux_qex = saturation_1(P, S1, smax)
        flux_u1 = split_1(phi, flux_qex)
        flux_q12 = split_1(1 - phi, flux_qex)
        flux_et = evap_16(gam, np.inf, S1, 0.01, Ep, delta_t)
        flux_u2 = saturation_9(flux_q12, S2, 0.01)
        flux_q = baseflow_1(k1, S3)

        # stores ODEs
        dS1 = P - flux_ea - flux_qex
        dS2 = flux_et + flux_u2 - flux_q12
        dS3 = flux_u1 + flux_u2 - flux_q

        # outputs
        dS = np.array([dS1, dS2, dS3])
        fluxes = np.array([flux_ea, flux_qex, flux_u1,
                           flux_q12, flux_et, flux_u2, flux_q])

        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep
        """
        pass
