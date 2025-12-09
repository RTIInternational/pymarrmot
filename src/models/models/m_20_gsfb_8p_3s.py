import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation import evap_20
from pymarrmot.models.flux.saturation import saturation_1
from pymarrmot.models.flux.interflow import interflow_11
from pymarrmot.models.flux.baseflow import baseflow_9, baseflow_1
from pymarrmot.models.flux.recharge import recharge_5

class m_20_gsfb_8p_3s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: GSFB

    Model reference:
    Nathan, R. J., & McMahon, T. A. (1990). SFB model part l. Validation of
    fixed model parameters. In Civil Eng. Trans. (pp. 157–161).

    Ye, W., Bates, B. C., Viney, N. R., & Sivapalan, M. (1997). Performance
    of conceptual rainfall-runoff models in low-yielding ephemeral catchments.
    Water Resources Research, 33(1), 153–166. http://doi.org/doi:10.1029/96WR02840
    """

    def __init__(self):
        """
        Creator method
        """
        super().__init__()
        self.num_stores = 3  # number of model stores
        self.num_fluxes = 6  # number of model fluxes
        self.num_params = 8  # number of model parameters

        self.jacob_pattern = np.array([[1, 0, 1],
                                      [1, 1, 0],
                                      [1, 1, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[0, 1],          # c, Recharge time coeffcient [d-1]
                                   [0.05, 0.95],    # ndc, Threshold fraction of Smax [-]
                                   [1, 2000],       # smax, Maximum soil moisture storage [mm]
                                   [0, 20],         # emax, Maximum evaporation flux [mm/d]
                                   [0, 200],        # frate, Maximum infiltration rate [mm/d]
                                   [0, 1],          # b, Fraction of subsurface flow that is baseflow [-]
                                   [0, 1],          # dpf, Baseflow time coefficient [d-1]
                                   [1, 300]])       # sdrmax, Threshold before baseflow can occur [mm]

        self.store_names = ["S1", "S2", "S3"]  # Names for the stores
        self.flux_names = ["ea", "qs", "f", "qb", "dp", "qdr"]  # Names for the fluxes

        self.flux_groups = {"Ea": 1,  # Index or indices of fluxes to add to Actual ET
                           "Q": [2, 4]}  # Index or indices of fluxes to add to Streamflow

    def init(self):
        """
        INITialisation function
        """
        pass

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation

        Parameters:
        S : array-like
            State variables

        Returns:
        tuple: Tuple containing two arrays (dS, fluxes)
               dS: Array of derivatives of state variables
               fluxes: Array of fluxes
        """
        # parameters
        theta = self.theta
        c, ndc, smax, emax, frate, b, dpf, sdrmax = theta

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
        flux_ea = evap_20(emax, ndc, S1, smax, Ep, delta_t)
        flux_qs = saturation_1(P, S1, smax)
        flux_f = interflow_11(frate, ndc * smax, S1, delta_t)
        flux_qb = baseflow_9(b * dpf, sdrmax, S2)
        flux_dp = baseflow_1((1 - b) * dpf, S2)
        flux_qdr = recharge_5(c, ndc * smax, S3, S1)

        # stores ODEs
        dS1 = P + flux_qdr - flux_ea - flux_qs - flux_f
        dS2 = flux_f - flux_qb - flux_dp
        dS3 = flux_dp - flux_qdr

        # outputs
        dS = np.array([dS1, dS2, dS3])
        fluxes = np.array([flux_ea, flux_qs, flux_f, flux_qb, flux_dp, flux_qdr])

        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep
        """
        pass
