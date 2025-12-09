import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.saturation import saturation_1, saturation_7
from pymarrmot.models.flux.evaporation import evap_3
from pymarrmot.models.flux.interflow import interflow_10
from pymarrmot.models.flux.baseflow import baseflow_4

class m_14_topmodel_7p_2s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: TOPMODEL

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model reference:
    Beven, K., Lamb, R., Quinn, P., Romanowicz, R., & Freer, J. (1995). 
    TOPMODEL. In V. P. Singh (Ed.), Computer Models of Watershed Hydrology 
    (pp. 627–668). Baton Rouge: Water Resources Publications, USA.

    Beven, K. J., & Kirkby, M. J. (1979). A physically based, variable 
    contributing area model of basin hydrology / Un modèle à base physique 
    de zone d’appel variable de l'hydrologie du bassin versant. Hydrological 
    Sciences Bulletin, 24(1), 43–69. http://doi.org/10.1080/02626667909491834

    Clark, M. P., Slater, A. G., Rupp, D. E., Woods, R. a., Vrugt, J. a., 
    Gupta, H. V., … Hay, L. E. (2008). Framework for Understanding Structural
    Errors (FUSE): A modular framework to diagnose differences between 
    hydrological models. Water Resources Research, 44(12). 
    http://doi.org/10.1029/2007WR006735

    Sivapalan, M., Beven, K., & Wood, E. F. (1987). On hydrologic similarity:
    2. A scaled model of storm runoff production. Water Resources Research, 
    23(12), 2266–2278. http://doi.org/10.1029/WR023i012p02266
    """

    def __init__(self, delta_t=None, theta=None):
        """
        creator method
        """
        super().__init__()
        self.num_stores = 2  # number of model stores
        self.num_fluxes = 6  # number of model fluxes
        self.num_params = 7
        
        self.jacob_pattern = np.array([[1, 1],
                                      [1, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[1, 2000],  # suzmax, Maximum soil moisture storage in unsaturated zone [mm]
                                    [0.05, 0.95],  # st, Threshold for flow generation and evap change as fraction of suzmax [-]
                                    [0, 1],  # kd, Leakage to saturated zone flow coefficient [mm/d]
                                    [0.1, 200],  # q0, Zero deficit base flow speed [mm/d]
                                    [0, 1],  # f, Baseflow scaling coefficient [mm-1]
                                    [1, 7.5],  # chi, Gamma distribution parameter [-]
                                    [0.1, 5]])  # phi, Gamma distribution parameter [-]
        
        self.store_names = ["S1", "S2"]  # Names for the stores
        self.flux_names = ["qof", "peff", "ea", "qex", "qv", "qb"]  # Names for the fluxes
        
        self.flux_groups = {"Ea": 3,   # Index or indices of fluxes to add to Actual ET
                           "Q": [1, 4, 6]}       # Index or indices of fluxes to add to Streamflow
        self.StoreSigns = [1, -1]  # Signs to give to stores (-1 is a deficit store), only needed for water balance

        # setting delta_t and theta triggers the function obj.init()
        if delta_t is not None:
            self.delta_t = delta_t
        if theta is not None:
            self.theta = theta

    def init(self):
        """
        INITialisation function
        """
        pass

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation

        Parameters:
            S (numpy.array): State variables

        Returns:
            tuple: Tuple containing the change in state variables and fluxes
        """
        # parameters
        theta = self.theta
        suzmax = theta[0]  # Maximum soil moisture storage in unsaturated zone [mm]
        st = theta[1]  # Threshold for flow generation and evap change as fraction of suzmax [-]
        kd = theta[2]  # Leakage to saturated zone flow coefficient [mm/d]
        q0 = theta[3]  # Zero deficit base flow speed [mm/d]
        f = theta[4]  # Baseflow scaling coefficient [mm-1]
        chi = theta[5]  # Gamma distribution parameter [-]
        phi = theta[6]  # Gamma distribution parameter [-]
        mu = 3  # Gamma distribution parameter, fixed (Clark et al, 2008)
        lambda_ = chi * phi + mu  # Ac computation parameter, mean of the gamma distribution

        # delta_t
        delta_t = self.delta_t
        
        # stores
        S1 = S[0]
        S2 = S[1]
        
        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]
        
        # fluxes functions
        flux_qof = saturation_7(chi, phi, 3, lambda_, f, S2, P)
        flux_peff = P - flux_qof
        flux_ea = evap_3(st, S1, suzmax, Ep, delta_t)
        flux_qex = saturation_1(flux_peff, S1, suzmax)
        flux_qv = interflow_10(S1, kd, st * suzmax, suzmax - st * suzmax)
        flux_qb = baseflow_4(q0, f, S2)

        # stores ODEs
        dS1 = flux_peff - flux_ea - flux_qex - flux_qv
        dS2 = flux_qb - flux_qv  # S2 is a deficit store

        # outputs
        dS = np.array([dS1, dS2])
        fluxes = np.array([flux_qof, flux_peff, flux_ea,
                           flux_qex, flux_qv, flux_qb])

        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep
        """
        pass

