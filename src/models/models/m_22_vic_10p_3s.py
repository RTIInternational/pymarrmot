import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.saturation import saturation_1, saturation_2
from pymarrmot.models.flux.evaporation import evap_7
from pymarrmot.models.flux.excess_1 import excess_1
from pymarrmot.models.flux.interception import interception_1
from pymarrmot.models.flux.percolation import percolation_5
from pymarrmot.models.flux.baseflow import baseflow_5
from pymarrmot.models.flux.effective_1 import effective_1
from pymarrmot.models.flux.phenology import phenology_2

class m_22_vic_10p_3s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: VIC

    References:
    - Clark, M. P., Slater, A. G., Rupp, D. E., Woods, R. A., Vrugt, J. A., Gupta, H. V., & Hay, L. E. (2008). 
      Framework for Understanding Structural Errors (FUSE): A modular framework to diagnose differences between
      hydrological models. Water Resources Research, 44(12), W00B02. http://doi.org/10.1029/2007WR006735
    - Liang, X., Lettenmaier, D. P., Wood, E. F., & Burges, S. J. (1994). A simple hydrologically based model of
      land surface water and energy fluxes for general circulation models. Journal of Geophysical Research, 99,
      14415–14428.
    """

    def __init__(self):
        super().__init__()
        self.aux_theta = None  # auxiliary parameter set

        # Model parameters
        self.num_stores = 3  # number of model stores
        self.num_fluxes = 11  # number of model fluxes
        self.num_params = 10

        self.jacob_pattern = np.array([[1, 0, 0],
                                      [1, 1, 0],
                                      [1, 1, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[0.1, 5],        # ibar, Mean interception capacity [mm]
                                    [0, 1],         # idelta, Seasonal interception change as fraction of mean [-]
                                    [1, 365],       # ishift, Maximum interception peak timing [-]
                                    [1, 2000],      # stot, Maximum soil moisture capacity [mm]
                                    [0.01, 0.99],   # fsm, Fraction of stot that constitutes maximum soil moisture smmax [-]
                                    [0, 10],        # b, Infiltration excess shape parameter [-]
                                    [0, 1],         # k1, Percolation time parameter [d-1]
                                    [0, 10],        # c1, Percolation non-linearity parameter [-]
                                    [0, 1],         # k2, Baseflow time parameter [d-1]
                                    [1, 5]])        # c2, Baseflow non-linearity parameter

        self.store_names = ["S1", "S2", "S3"]  # Names for the stores
        self.flux_names = ["ei", "peff", "iex", "qie", "inf", "et1",
                          "qex1", "pc", "et2", "qex2", "qb"]  # Names for the fluxes

        self.flux_groups = {"Ea": [1, 6, 9],  # Index or indices of fluxes to add to Actual ET
                           "Q": [4, 7, 10, 11]}  # Index or indices of fluxes to add to Streamflow

    def init(self):
        """
        Initialization function.
        """
        # Parameters
        theta = self.theta
        stot = theta[3]  # Total available storage [mm]
        fsm = theta[4]  # Fraction of stot that constitutes maximum soil moisture storage [-]

        # Auxiliary parameter
        smmax = fsm * stot  # Maximum soil moisture capacity [mm]
        gwmax = (1 - fsm) * stot  # Maximum groundwater storage [mm]
        tmax = 365.25  # Length of one growing cycle [d]
        self.aux_theta = np.array([smmax, gwmax, tmax])

    def model_fun(self, S):
        """
        Model governing equations in state-space formulation.

        Parameters:
        - S : numpy.ndarray
            State variables.

        Returns:
        - dS : numpy.ndarray
            State derivatives.
        - fluxes : numpy.ndarray
            Model fluxes.
        """
        # Parameters
        theta = self.theta
        ibar, idelta, ishift, _, _, b, k1, c1, k2, c2 = theta

        # Auxiliary parameters
        smmax, gwmax, tmax = self.aux_theta

        # Delta_t
        delta_t = self.delta_t

        # Stores
        S1, S2, S3 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        # Fluxes functions
        aux_imax = phenology_2(ibar, idelta, ishift, self.t+1, tmax, delta_t)
        flux_ei = evap_7(S1, aux_imax, Ep, delta_t)
        flux_peff = interception_1(P, S1, aux_imax)
        flux_iex = excess_1(S1, aux_imax, delta_t)
        flux_qie = saturation_2(S2, smmax, b, flux_peff + flux_iex)
        flux_inf = effective_1(flux_peff + flux_iex, flux_qie)
        flux_et1 = evap_7(S2, smmax, max(0, Ep - flux_ei), delta_t)
        flux_qex1 = saturation_1(flux_inf, S2, smmax)
        flux_pc = percolation_5(k1, c1, S2, smmax, delta_t)
        flux_et2 = evap_7(S3, gwmax, max(0, Ep - flux_ei - flux_et1), delta_t)
        flux_qex2 = saturation_1(flux_pc, S3, gwmax)
        flux_qb = baseflow_5(k2, c2, S3, gwmax, delta_t)

        # Stores ODEs
        dS1 = P - flux_ei - flux_peff - flux_iex
        dS2 = flux_inf - flux_et1 - flux_qex1 - flux_pc
        dS3 = flux_pc - flux_et2 - flux_qex2 - flux_qb

        # Outputs
        dS = np.array([dS1, dS2, dS3])
        fluxes = np.array([flux_ei, flux_peff, flux_iex, flux_qie,
                           flux_inf, flux_et1, flux_qex1, flux_pc,
                           flux_et2, flux_qex2, flux_qb])

        return dS, fluxes

    def step(self):
        """
        Runs at the end of every timestep.
        """
        pass
