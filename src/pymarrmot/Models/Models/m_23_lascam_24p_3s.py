import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation import evap_1, evap_19
from pymarrmot.models.flux.area_1 import area_1
from pymarrmot.models.flux.infiltration import infiltration_4, infiltration_5
from pymarrmot.models.flux.interception import interception_5
from pymarrmot.models.flux.recharge import recharge_3, recharge_4
from pymarrmot.models.flux.effective_1 import effective_1
from pymarrmot.models.flux.saturation import saturation_11, saturation_12
from pymarrmot.models.flux.baseflow import baseflow_8

class M_23_LASCAM_24p_3s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Large-scale catchment water and salt balance model element

    References:
    - Sivapalan, M., Ruprecht, J. K., & Viney, N. R. (1996). Water and salt balance modelling to predict the effects 
      of land-use changes in forested catchments. 1. Small catchment water balance model. Hydrological Processes, 10(3).
    """

    def __init__(self):
        super().__init__()
        self.aux_theta = None  # auxiliary parameters

        # Model parameters
        self.numStores = 3  # number of model stores
        self.numFluxes = 16  # number of model fluxes
        self.numParams = 24

        self.JacobPattern = np.ones((3, 3), dtype=int)  # Jacobian matrix of model store ODEs

        self.parRanges = np.array([[0, 200],        # af, Catchment-scale infiltration parameter [mm/d]
                                    [0, 5],         # bf, Catchment-scale infiltration non-linearity parameter [-]
                                    [1, 2000],      # stot, Total catchment storage [mm]
                                    [0.01, 0.99],   # xa, Fraction of Stot that is Amax [-]
                                    [0.01, 0.99],   # xf, Fraction of Stot-Amx that is depth Fmax [-]
                                    [0.01, 0.99],   # na, Fraction of Amax that is Amin [-]
                                    [0, 5],         # ac, Variable contributing area scaling [-]
                                    [0, 10],        # bc, Variable contributing area non-linearity [-]
                                    [0, 5],         # ass, Subsurface saturation area scaling [-]
                                    [0, 10],        # bss, Subsurface saturation area non-linearity [-]
                                    [0, 200],       # c, Maximum infiltration rate [mm/d]
                                    [0, 5],         # ag, Interception base parameter [mm/d]
                                    [0, 1],         # bg, Interception fraction parameter [-]
                                    [0, 1],         # gf, F-store evaporation scaling [-]
                                    [0, 10],        # df, F-store evaporation non-linearity [-]
                                    [0, 1],         # rd, Recharge time parameter [d-1]
                                    [0, 1],         # ab, Groundwater flow scaling [-]
                                    [0.01, 200],    # bb, Groundwater flow base rate [mm/d]
                                    [0, 1],         # ga, A-store evaporation scaling [-]
                                    [0, 10],        # da, A-store evaporation non-linearity [-]
                                    [0.01, 200],    # aa, Subsurface storm flow rate [mm/d]
                                    [1, 5],         # ba, Subsurface storm flow non-linearity [-]
                                    [0, 1],         # gb, B-store evaporation scaling [-]
                                    [0, 10]])       # db, B-store evaporation non-linearity [-]

        self.StoreNames = ["S1", "S2", "S3"]  # Names for the stores
        self.FluxNames = ["ei", "pg", "qse", "qie", "pc",
                          "qsse", "qsie", "fa", "ef", "rf",
                          "ea1", "ea2", "qa", "ra", "qb", "eb"]  # Names for the fluxes

        self.FluxGroups = {"Ea": [1, 9, 11, 12, 16],  # Index or indices of fluxes to add to Actual ET
                           "Q": [3, 4, 13]}  # Index or indices of fluxes to add to Streamflow

    def init(self):
        """
        Initialization function.
        """
        # Parameters
        theta = self.theta
        stot = theta[2]  # Total catchment storage [mm]
        xa = theta[3]  # Fraction of Stot that is Amax [-]
        xf = theta[4]  # Fraction of Stot-Amx that is depth Fmax [-]
        na = theta[5]  # Fraction of Amax that is Amin [-]

        # Auxiliary parameters
        amax = xa * stot  # Maximum contributing area depth [mm]
        fmax = xf * (stot - amax)  # Infiltration depth scaling [mm]
        bmax = (1 - xf) * (stot - amax)  # Groundwater depth scaling [mm]
        amin = na * amax  # Minimum contributing area depth [mm]
        self.aux_theta = np.array([amax, fmax, bmax, amin])

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
        af, bf, _, _, _, ac, bc, ass, bss, c, ag, bg, gf, df, td, ab, bb, ga, da, aa, ba, gb, db = theta

        # Auxiliary parameters
        aux_theta = self.aux_theta
        amax, _, bmax, _ = aux_theta

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
        tmp_phiss = area_1(ass, bss, S2, aux_theta[3], amax)
        tmp_phic = area_1(ac, bc, S2, aux_theta[3], amax)
        tmp_fss = infiltration_5(af, bf, S3, bmax, S1, aux_theta[1])

        flux_pg = interception_5(bg, ag, P)
        flux_ei = effective_1(P, flux_pg)
        flux_qse = saturation_11(ac, bc, S2, aux_theta[3], amax, flux_pg)
        flux_pc = infiltration_4(flux_pg - flux_qse, c)
        flux_qie = effective_1(flux_pg - flux_qse, flux_pc)
        flux_qsse = saturation_12(tmp_phiss, tmp_phic, flux_pc)
        flux_fa = infiltration_4(max(0, flux_pc * min(1, (1 - tmp_phiss) / (1 - tmp_phic))), tmp_fss)
        flux_qsie = effective_1(flux_pc, flux_fa + flux_qsse)
        flux_fa = infiltration_4(max(0, flux_pc * min(1, (1 - tmp_phiss) / (1 - tmp_phic))), tmp_fss)
        flux_qsie = effective_1(flux_pc, flux_fa + flux_qsse)
        flux_ef = evap_19(gf, df, S1, aux_theta[1], Ep, delta_t)
        flux_rf = recharge_3(td, S1)
        flux_ea1 = evap_1(S2, tmp_phic * Ep, delta_t)
        flux_ea2 = evap_19(ga, da, S2, amax, Ep, delta_t)
        flux_qa = saturation_11(aa, ba, S2, aux_theta[3], amax, 1)
        flux_ra = recharge_4(tmp_phic, tmp_fss, delta_t)
        flux_qb = baseflow_8(bb, ab, S3, bmax)
        flux_eb = evap_19(gb, db, S3, bmax, Ep, delta_t)

        # Stores ODEs
        dS1 = flux_fa - flux_ef - flux_rf
        dS2 = flux_qsse + flux_qsie + flux_qb - flux_ea1 - flux_ea2 - flux_ra - flux_qa
        dS3 = flux_rf + flux_ra - flux_eb - flux_qb

        # Outputs
        dS = np.array([dS1, dS2, dS3])
        fluxes = np.array([flux_ei, flux_pg, flux_qse, flux_qie,
                           flux_pc, flux_qsse, flux_qsie, flux_fa,
                           flux_ef, flux_rf, flux_ea1, flux_ea2,
                           flux_qa, flux_ra, flux_qb, flux_eb])

        return dS, fluxes

    def step(self):
        """
        Runs at the end of every timestep.
        """
        pass

