import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation.evap_5 import evap_5
from pymarrmot.models.flux.evaporation.evap_8 import evap_8
from pymarrmot.models.flux.evaporation.evap_9 import evap_9
from pymarrmot.models.flux.evaporation.evap_10 import evap_10
from pymarrmot.models.flux.saturation.saturation_1 import saturation_1
from pymarrmot.models.flux.interception.interception_3 import interception_3
from pymarrmot.models.flux.excess_1 import excess_1
from pymarrmot.models.flux.baseflow.baseflow_1 import baseflow_1

class m_08_us1_5p_2s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: United States model v1
    """

    def __init__(self):
        super().__init__()
        self.num_stores = 2  # number of model stores
        self.num_fluxes = 9  # number of model fluxes
        self.num_params = 5

        self.jacob_pattern = np.array([[1, 1], [1, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[0, 1],            # Alpha_ei, Fraction of intercepted rainfall [-]
                                     [0.05, 0.95],     # M, Fraction forest [-]
                                     [1, 2000],        # Smax, Maximum soil moisture [mm]
                                     [0.05, 0.95],     # fc, Field capacity as fraction of smax [-]
                                     [0, 1]])          # Alpha_ss, Subsurface routing delay [d-1]

        self.store_names = ["S1", "S2"]  # Names for the stores
        self.flux_names = ["eusei", "eusveg", "eusbs", "esatveg",
                           "esatbs", "rg", "se", "qse", "qss"]  # Names for the fluxes

        self.flux_groups = {"Ea": [1, 2, 3, 4, 5],  # Index or indices of fluxes to add to Actual ET
                            "Q": [8, 9]}            # Index or indices of fluxes to add to Streamflow

    def init(self):
        pass

    def model_fun(self, S):
        # parameters
        alpha_ei, m, smax, fc, alpha_ss = self.theta

        # delta_t
        delta_t = self.delta_t

        # stores
        S1, S2 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        # fluxes functions
        flux_eusei = interception_3(alpha_ei, P)
        flux_eusveg = evap_8(S1, S2, m, fc * (smax - S2), Ep, delta_t)
        flux_eusbs = evap_9(S1, S2, m, smax, Ep, delta_t)
        # BUG: third argument for evap_10 and evap_5 was changed from S1 + S2 
        # (in Matlab to Python translation) to smax 
        # which is what is expected in the functions
        flux_esatveg = evap_10(m, S2, smax, Ep, delta_t)
        flux_esatbs = evap_5(m, S2, smax, Ep, delta_t)
        flux_rg = saturation_1(P, S1, fc * (smax - S2))
        flux_se = excess_1(S1, fc * (smax - S2), delta_t)
        flux_qse = saturation_1(flux_rg + flux_se, S2, smax)
        flux_qss = baseflow_1(alpha_ss, S2)

        # stores ODEs
        dS1 = P - flux_eusei - flux_eusveg - flux_eusbs - flux_rg - flux_se
        dS2 = flux_rg + flux_se - flux_esatveg - flux_esatbs - flux_qse - flux_qss

        # outputs
        dS = np.array([dS1, dS2])
        fluxes = [flux_eusei, flux_eusveg, flux_eusbs,
                  flux_esatveg, flux_esatbs, flux_rg,
                  flux_se, flux_qse, flux_qss]

        return dS, fluxes

    def step(self):
        pass
