from marrmot_model import MARRMoT_model
from ..Flux import interception_2, evap_1, saturation_2, baseflow_1

class M_02_Wetland_4p_1s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Wetland model
    Model reference:
    Savenije, H. H. G. (2010). “Topography driven conceptual modelling (FLEX-Topo).” Hydrology and Earth System Sciences, 14(12), 2681–2692. https://doi.org/10.5194/hess-14-2681-2010
    """
    def __init__(self):
        self.num_stores = 1  # number of model stores
        self.num_fluxes = 5  # number of model fluxes
        self.num_params = 4
        self.jacob_pattern = [1]  # Jacobian matrix of model store ODEs
        self.par_ranges = [[0, 5],  # Dw, interception capacity [mm]
                           [0, 10],  # Betaw, soil misture distribution parameter [-]
                           [1, 2000],  # Swmax, soil misture depth [mm]
                           [0, 1]]  # kw, base flow time parameter [d-1]
        self.store_names = ["S1"]  # Names for the stores
        self.flux_names = ["pe", "ei", "ew", "qwsof", "qwgw"]  # Names for the fluxes
        self.flux_groups = {'Ea': [2, 3],  # Index or indices of fluxes to add to Actual ET
                            'Q': [4, 5]}  # Index or indices of fluxes to add to Streamflow

    def init(self):
        pass

    def model_fun(self, S):
        # parameters
        theta = self.theta
        dw = theta[0]  # Daily interception [mm]
        betaw = theta[1]  # Soil moisture storage distribution parameter [-]
        swmax = theta[2]  # Maximum soil moisture storage [mm]
        kw = theta[3]  # Runoff coefficient [d-1]
        # delta_t
        delta_t = self.delta_t
        # stores
        S1 = S[0]
        # climate input
        t = self.t  # this time step
        climate_in = self.input_climate[t, :]  # climate at this step
        P = climate_in[0]
        Ep = climate_in[1]
        T = climate_in[2]
        # fluxes functions
        flux_pe = interception_2(P, dw)
        flux_ei = P - flux_pe
        flux_ew = evap_1(S1, Ep, delta_t)
        flux_qwsof = saturation_2(S1, swmax, betaw, flux_pe)
        flux_qwgw = baseflow_1(kw, S1)
        # stores ODEs
        dS1 = flux_pe - flux_ew - flux_qwsof - flux_qwgw
        # outputs
        dS = [dS1]
        fluxes = [flux_pe, flux_ei, flux_ew, flux_qwsof, flux_qwgw]
        return dS, fluxes

    def step(self):
        pass
