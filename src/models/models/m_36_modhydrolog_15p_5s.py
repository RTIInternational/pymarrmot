import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation import evap_1, evap_2
from pymarrmot.models.flux.interception import interception_1
from pymarrmot.models.flux.infiltration import(infiltration_1, infiltration_2)
from pymarrmot.models.flux.interflow import interflow_1
from pymarrmot.models.flux.recharge import recharge_1
from pymarrmot.models.flux.saturation import saturation_1
from pymarrmot.models.flux.depression_1 import depression_1
from pymarrmot.models.flux.baseflow import baseflow_1
from pymarrmot.models.flux.exchange import(exchange_3, exchange_1)

class m_36_modhydrolog_15p_5s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: MODHYDROLOG

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model references:
    Chiew, F. H. S. (1990). Estimating groundwater recharge using an
    integrated surface and groundwater model. University of Melbourne.
    Chiew, F., & McMahon, T. (1994). Application of the daily rainfall-runoff
    model MODHYDROLOG to 28 Australian catchments. Journal of Hydrology,
    153(1–4), 383–416. https://doi.org/10.1016/0022-1694(94)90200-3
    """

    def __init__(self):
        """
        Creator method.
        """
        super().__init__()
        self.num_stores = 5  # number of model stores
        self.num_fluxes = 16  # number of model fluxes
        self.num_params = 15

        # Jacobian matrix of model store ODEs
        self.jacob_pattern = [[1, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0],
                              [1, 1, 1, 0, 0],
                              [1, 1, 1, 1, 0],
                              [1, 1, 1, 1, 1]]

        self.par_ranges = [[0, 5],     # INSC, Maximum interception capacity, [mm]
                           [0, 600],   # COEFF, Maximum infiltration loss parameter, [mm]
                           [0, 15],    # SQ, Infiltration loss exponent, [-]
                           [1, 2000],  # SMSC, Maximum soil moisture capacity, [mm]
                           [0, 1],     # SUB, Proportionality constant, [-]
                           [0, 1],     # CRAK, Proportionality constant, [-]
                           [0, 20],    # EM, maximum plant-controled evap rate, [mm/d]
                           [0, 50],    # DSC, Maximum depression capacity, [mm]
                           [0, 1],     # ADS, Land fraction functioning as depression storage, [-]
                           [0.99, 1],  # MD, Depression storage parameter, [-]
                           [0, 0.5],   # VCOND, Leakage coefficient, [mm/d]
                           [-10, 10],  # DLEV, Datum around which groundwater fluctuates relative to river bed, [mm]
                           [0, 1],     # K1, Flow exchange parameter, [d-1]
                           [0, 1],     # K2, Flow exchange parameter, [d-1]
                           [0, 100]]   # K3, Flow exchange parameter, [d-1]

        self.store_names = ["S1", "S2", "S3", "S4", "S5"]  # Names for the stores
        self.flux_names = ["Ei", "EXC", "INF", "INT",
                           "REC", "SMF", "Et", "GWF",
                           "TRAP", "Ed", "DINF", "SEEP",
                           "FLOW", "Q", "RUN", "SRUN"]  # Names for the fluxes

        self.flux_groups = {"Ea": [1, 7, 10],  # Index or indices of fluxes to add to Actual ET
                            "Q": [14],       # Index or indices of fluxes to add to Streamflow
                            "GWseepage": 12}  # Index or GW seepage flux

    def init(self):
        """
        Initialization function.
        """
        self.store_min[3] = float('-inf')

    def model_fun(self, S):
        """
        Model governing equations in state-space formulation.

        Parameters:
        -----------
        S : array_like
            State variables.

        Returns:
        --------
        dS : array_like
            Time derivatives of the state variables.
        fluxes : array_like
            Fluxes.
        """
        theta = self.theta
        insc = theta[0]     # Maximum interception capacity, [mm]
        coeff = theta[1]    # Maximum infiltration loss parameter, [-]
        sq = theta[2]       # Infiltration loss exponent, [-]
        smsc = theta[3]     # Maximum soil moisture capacity, [mm]
        sub = theta[4]      # Proportionality constant, [-]
        crak = theta[5]     # Proportionality constant, [-]
        em = theta[6]       # Plant-controled maximum evaporation rate [mm/d]
        dsc = theta[7]      # Maximum depression capacity, [mm]
        ads = theta[8]      # Land fraction functioning as depression storage, [-]
        md = theta[9]       # Depression storage parameter, [-], default = 1
        vcond = theta[10]   # Leakage coefficient, [mm/d]
        dlev = theta[11]    # Datum around which groundwater fluctuates relative to river bed, [mm]
        k1 = theta[12]      # Flow exchange parameter, [d-1]
        k2 = theta[13]      # Flow exchange parameter, [d-1]
        k3 = theta[14]      # Flow exchange parameter, [d-1]

        # delta_t
        delta_t = self.delta_t

        # stores
        S1, S2, S3, S4, S5 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        # Fluxes functions
        flux_Ei = evap_1(S1, Ep, delta_t)
        flux_EXC = interception_1(P, S1, insc)
        flux_INF = infiltration_1(coeff, sq, S2, smsc, flux_EXC)
        flux_INT = interflow_1(sub, S2, smsc, flux_INF)
        flux_REC = recharge_1(crak, S2, smsc, flux_INF - flux_INT)
        flux_SMF = flux_INF - flux_INT - flux_REC
        flux_Et = evap_2(em, S2, smsc, Ep, delta_t)
        flux_GWF = saturation_1(flux_SMF, S2, smsc)
        flux_RUN = flux_EXC - flux_INF
        flux_TRAP = depression_1(ads, md, S3, dsc, flux_RUN, delta_t)
        flux_Ed = evap_1(S3, ads * Ep, delta_t)
        flux_DINF = ads * infiltration_2(coeff, sq, S2, smsc, flux_SMF, S3, delta_t)
        flux_SEEP = exchange_3(vcond, S4, dlev)
        flux_SRUN = flux_RUN - flux_TRAP
        flux_FLOW = exchange_1(k1, k2, k3, S4, flux_SRUN, delta_t)
        flux_Q = baseflow_1(1, S5)

        # stores ODEs
        dS1 = P - flux_Ei - flux_EXC
        dS2 = flux_SMF + flux_DINF - flux_Et - flux_GWF
        dS3 = flux_TRAP - flux_Ed - flux_DINF
        dS4 = flux_REC + flux_GWF - flux_SEEP - flux_FLOW
        dS5 = flux_SRUN + flux_INT + flux_FLOW - flux_Q

        # outputs
        dS = np.array([dS1, dS2, dS3, dS4, dS5])
        fluxes = [flux_Ei, flux_EXC, flux_INF, flux_INT,
                  flux_REC, flux_SMF, flux_Et, flux_GWF,
                  flux_TRAP, flux_Ed, flux_DINF, flux_SEEP,
                  flux_FLOW, flux_Q, flux_RUN, flux_SRUN]

        return dS, fluxes

    def step(self):
        """
        Runs at the end of every timestep.
        """
        pass

