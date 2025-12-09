import numpy as np
from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.saturation.saturation_4 import saturation_4
from pymarrmot.models.flux.evaporation.evap_11 import evap_11
from pymarrmot.models.flux.percolation.percolation_3 import percolation_3
from pymarrmot.models.flux.recharge.recharge_2 import recharge_2
from pymarrmot.models.flux.baseflow.baseflow_3 import baseflow_3
from pymarrmot.models.unit_hydro.uh_1_half import uh_1_half
from pymarrmot.models.unit_hydro.uh_2_full import uh_2_full
from pymarrmot.models.unit_hydro.route import route
from pymarrmot.models.unit_hydro.update_uh import update_uh

class m_07_gr4j_4p_2s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: GR4J
    """

    def __init__(self):
        super().__init__()
        self.num_stores = 2  # number of model stores
        self.num_fluxes = 13  # number of model fluxes
        self.num_params = 4

        self.jacob_pattern = np.array([[1, 1],
                                        [1, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[1, 2000],    # x1 [mm]
                                     [-20, 20],   # x2 [mm/d]
                                     [1, 300],    # x3 [mm]
                                     [0.5, 15]])  # x4 [d]

        self.store_names = ["S1", "S2"]  # Names for the stores
        self.flux_names = ["pn", "en", "ef", "ps", "es", "perc",
                           "q9", "q1", "fr", "fq", "qr", "qt", "ex"]  # Names for the fluxes

        self.flux_groups = {"Ea": [3, 5],      # Index or indices of fluxes to add to Actual ET
                            "Q": [12],         # Index or indices of fluxes to add to Streamflow
                            "Exchange": -13}  # Index or indices of exchange fluxes

    def init(self):
        """
        INITialisation function
        """
        ##super().__init__()
        theta = self.theta
        delta_t = self.delta_t
        x1 = theta[0]  # Maximum soil moisture storage [mm]
        x3 = theta[2]  # Maximum routing store storage [mm]
        x4 = theta[3]  # Flow delay [d]

        # max of stores
        #self.store_max = [x1, x3]
        self.store_max = np.array([x1, x3])

        # initialise the unit hydrographs and still-to-flow vectors
        uh_q9 = uh_1_half(x4, delta_t)
        uh_q1 = uh_2_full(2 * x4, delta_t)

        self.uhs = [uh_q9, uh_q1]

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation

        Parameters:
        - S: array-like, current state variables

        Returns:
        - dS: array-like, model state derivatives
        - fluxes: array-like, model fluxes
        """
        theta = self.theta
        x1, x2, x3, x4 = theta

        # delta_t
        delta_t = self.delta_t

        # unit hydrographs and still-to-flow vectors
        uh_q9, uh_q1 = self.uhs

        # stores
        S1, S2 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]
        
        # fluxes functions
        flux_pn = max(P - Ep, 0)
        flux_en = max(Ep - P, 0)
        flux_ef = P - flux_pn
        flux_ps = saturation_4(S1, x1, flux_pn)
        flux_es = evap_11(S1, x1, flux_en)
        flux_perc = percolation_3(S1, x1)
        flux_q9 = route(0.9 * (flux_pn - flux_ps + flux_perc), uh_q9)
        flux_q1 = route(0.1 * (flux_pn - flux_ps + flux_perc), uh_q1)
        flux_fr = recharge_2(3.5, S2, x3, x2)
        flux_fq = flux_fr
        flux_qr = baseflow_3(S2, x3)
        flux_qt = flux_qr + max(flux_q1 + flux_fq, 0)
        flux_ex = flux_fr + max(flux_q1 + flux_fq, 0) - flux_q1

        # stores ODEs
        dS1 = flux_ps - flux_es - flux_perc
        dS2 = flux_q9 + flux_fr - flux_qr

        # outputs
        dS = np.array([dS1, dS2])
        fluxes = [flux_pn, flux_en, flux_ef, flux_ps, flux_es,
                  flux_perc, flux_q9, flux_q1, flux_fr, flux_fq,
                  flux_qr, flux_qt, flux_ex]

        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep.
        """
        uh_q9, uh_q1 = self.uhs

        # input fluxes to the unit hydrographs
        fluxes = self.fluxes[self.t, :]
        flux_pn, flux_ps, flux_perc = fluxes[0], fluxes[3], fluxes[5]

        # update still-to-flow vectors using fluxes at current step and
        # unit hydrographs
        uh_q9 = update_uh(uh_q9, 0.9 * (flux_pn - flux_ps + flux_perc))
        uh_q1 = update_uh(uh_q1, 0.1 * (flux_pn - flux_ps + flux_perc))

        self.uhs = [uh_q9, uh_q1]
