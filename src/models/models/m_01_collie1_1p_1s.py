import numpy as np
from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation.evap_7 import evap_7
from pymarrmot.models.flux.saturation.saturation_1 import saturation_1

class m_01_collie1_1p_1s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Collie River 1 (traditional bucket model)
    Model reference:
    Jothityangkoon, C., M. Sivapalan, and D. Farmer (2001), Process controls
    of water balance variability in a large semi-arid catchment: downward 
    approach to hydrological model development. Journal of Hydrology, 254,
    174–198. doi: 10.1016/S0022-1694(01)497-6.
    """
    def __init__(self):
        super().__init__()
        self.num_stores = 1                  # number of model stores
        self.num_fluxes = 2                  # number of model fluxes
        self.num_params = 1
        self.jacob_pattern = [1]             # Jacobian matrix of model store ODEs
        self.par_ranges = np.array([[1, 2000]])   # Smax, Maximum soil moisture storage [mm]
        self.store_names = ["S1"]            # Names for the stores
        self.flux_names = ["ea", "qse"]      # Names for the fluxes
        self.flux_groups = {'Ea': 1, 'Q': 2} # Index or indices of fluxes to add to Actual ET and Streamflow

    def init(self):
        pass

    def model_fun(self, S):
        # parameters
        theta = self.theta
        S1max = theta[0]  # Maximum soil moisture storage [mm]

        delta_t = self.delta_t
        
        # stores
        S1 = S[0]
        
        # climate input
        t = self.t  # this time step
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        
        # fluxes functions
        flux_ea = evap_7(S1, S1max, Ep, delta_t)
        flux_qse = saturation_1(P, S1, S1max)
        
        # stores ODEs
        dS1 = P - flux_ea - flux_qse
        
        # outputs
        dS = np.array([dS1])
        fluxes = [flux_ea, flux_qse]
        return dS, fluxes

    def step(self):
        pass


