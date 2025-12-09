import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation import evap_21
from pymarrmot.models.flux.saturation import saturation_14
from pymarrmot.models.flux.split import split_1
from pymarrmot.models.flux.saturation import saturation_2
from pymarrmot.models.flux.interflow import interflow_5
from pymarrmot.models.flux.baseflow import baseflow_1

class m_28_xinanjiang_12p_4s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Xinanjiang
    """

    def __init__(self):
        """
        Creator method.
        """
        super().__init__()
        self.num_stores = 4                                          # number of model stores
        self.num_fluxes = 10                                         # number of model fluxes
        self.num_params = 12

        self.jacob_pattern = np.array([[1, 0, 0, 0],
                                       [1, 1, 0, 0],
                                       [0, 1, 1, 0],
                                       [0, 1, 0, 1]])              # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[0, 1],                           # aim,  Fraction impervious area [-]
                                   [-0.49, 0.49],                   # a,    Tension water distribution inflection parameter [-]
                                   [0, 10],                         # b,    Tension water distribution shape parameter [-]
                                   [1, 2000],                       # stot, Total soil moisture storage (W+S) [mm]
                                   [0.01, 0.99],                    # fwm,  Fraction of Stot that is Wmax [-]
                                   [0.01, 0.99],                    # flm,  Fraction of wmax that is LM [-]
                                   [0.01, 0.99],                    # c,    Fraction of LM for second evaporation change [-]
                                   [0, 10],                         # ex,   Free water distribution shape parameter [-]
                                   [0, 1],                          # ki,   Free water interflow parameter [d-1]
                                   [0, 1],                          # kg,   Free water groundwater parameter [d-1]
                                   [0, 1],                          # ci,   Interflow time coefficient [d-1]
                                   [0, 1]])                         # cg,   Baseflow time coefficient [d-1]

        self.store_names = ["S1", "S2", "S3", "S4"]                  # Names for the stores
        self.flux_names = ["rb", "pi", "e", "r", "rs",
                          "ri", "rg", "qs", "qi", "qg"]            # Names for the fluxes

        self.flux_groups = {'Ea': 3, 'Q': [8, 9, 10]}                # Index or indices of fluxes to add to Actual ET and Streamflow

        self.aux_theta = None

    def init(self):
        """
        Initialization function.
        """
        # parameters
        aim, a, b, stot, fwm, flm = self.theta[:6]                   # Unpack theta
        wmax = fwm * stot                                            # Maximum tension water depth [mm]
        smax = (1 - fwm) * stot                                      # Maximum free water depth [mm]
        lm = flm * wmax                                              # Tension water threshold for evaporation change [mm]
        self.aux_theta = [wmax, smax, lm]

    def model_fun(self, S):
        """
        Model governing equations in state-space formulation.
        """
        # parameters
        aim, a, b, _, _, _, c, ex, ki, kg, ci, cg = self.theta      # Unpack theta

        # auxiliary parameters
        wmax, smax, lm = self.aux_theta

        # delta_t
        delta_t = self.delta_t

        # stores
        S1, S2, S3, S4 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        # fluxes functions
        flux_rb = split_1(aim, P)
        flux_pi = split_1(1 - aim, P)
        flux_e = evap_21(lm, c, S1, Ep, delta_t)
        flux_r = saturation_14(a, b, S1, wmax, flux_pi)
        flux_rs = saturation_2(S2, smax, ex, flux_r)
        flux_ri = saturation_2(S2, smax, ex, S2 * ki)
        flux_rg = saturation_2(S2, smax, ex, S2 * kg)
        flux_qs = flux_rb + flux_rs
        flux_qi = interflow_5(ci, S3)
        flux_qg = baseflow_1(cg, S4)

        # stores ODEs
        dS1 = flux_pi - flux_e - flux_r
        dS2 = flux_r - flux_rs - flux_ri - flux_rg
        dS3 = flux_ri - flux_qi
        dS4 = flux_rg - flux_qg

        # outputs
        dS = np.array([dS1, dS2, dS3, dS4])
        fluxes = np.array([flux_rb, flux_pi, flux_e, flux_r, flux_rs,
                           flux_ri, flux_rg, flux_qs, flux_qi, flux_qg])

        return dS, fluxes

    def step(self):
        """
        Runs at the end of every timestep.
        """
        pass
