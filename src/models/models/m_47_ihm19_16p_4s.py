import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation import evap_1, evap_23
from pymarrmot.models.flux.interception import interception_1
from pymarrmot.models.flux.split import split_1, split_2
from pymarrmot.models.flux.infiltration import (infiltration_3, infiltration_7)
from pymarrmot.models.flux.interflow import interflow_3, interflow_9
from pymarrmot.models.unit_hydro import (route, uh_4_full, update_uh)

class m_47_ihm19_16p_4s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Forellenbach model
    
    Copyright (C) 2021 Clara Brandes, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.
    
    Model reference:
    Brandes, C. (2020). Erstellung eines Konzeptionellen
    Hochwasserabfluss-modells für das Einzugsgebiet des Forellenbachs, NP
    Bayerischer Wald. MSc Thesis. Technische Univerisität Dresden, Germany.
    """

    def __init__(self):
        super().__init__()
        self.num_stores = 4  # number of model stores
        self.num_fluxes = 18  # number of model fluxes
        self.num_params = 16

        self.jacob_pattern = np.array([[1, 0, 0, 0],
                                      [1, 1, 0, 0],
                                      [1, 1, 1, 0],
                                      [0, 1, 1, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[2, 5],         # 00 SIMAX, maximum interception storage [mm]
                                    [0.9, 1],       # 01 A, splitting coefficient for excess precipitation [-]
                                    [0.4, 0.95],    # 02 FF, forest fraction [-]
                                    [0.05, 5],      # 03 SMPMAX, maximum storage macropores [mm]
                                    [0, 1],         # 04 CQMP, runoff time parameter (fast/slow runoff) first soil layer [1/d]
                                    [1, 5],         # 05 XQMP, runoff scale parameter first soil layer [-]
                                    [400, 600],     # 06 SS1MAX, maximum soil moisture storage first soil layer [mm]
                                    [0.3, 0.7],     # 07 FCCS1, field capacity as fraction of maximum storage first soil layer [-]
                                    [0, 1000],      # 08 CFS1, maximum infiltration rate first soil layer [mm/d]
                                    [0, 15],        # 09 XFS1, infiltration loss exponent first soil layer [-]
                                    [0, 1],         # 10 CQS1, runoff time parameter for (fast/slow runoff) first soil layer [1/d]
                                    [1, 5],         # 11 XQS1, runoff scale parameter first soil layer [-]
                                    [300, 500],     # 12 SS2MAX, maximum soil moisture storage second soil layer [mm]
                                    [0, 1],         # 13 CQS2, runoff time parameter for (fast/slow runoff) second soil layer [1/d]
                                    [1, 5],         # 14 XQS2, runoff scale parameter second soil layer [-]
                                    [0.01, 5]])     # 15 D, Flow delay before surface runoff [d]

        self.store_names = ["S1", "S2", "S3", "S4"] # Names for the stores
        self.flux_names = ["ei", "pex", "pexmp", "pexs1", "fmp",
                          "qexmp", "qmp", "pqexsl", "fs1", "etas1",
                          "qs1", "q0", "q0r", "qmps1", "pc",
                          "qh", "qs2", "qgw"]       # Names for the fluxes

        self.flux_groups = {"Ea": [1, 10],          # Index or indices of fluxes to add to Actual ET
                           "Q": [13, 16, 17, 18],   # Index or indices of fluxes to add to Streamflow
                           "GW": -18}               # Index of GW runoff flow.

    def init(self):
        """
        Initialisation function.
        """
        theta = self.theta
        delta_t = self.delta_t
        SIMAX = theta[0]  # maximum interception storage [mm]
        SMPMAX = theta[3]  # maximum storage macropores [mm]
        SS1MAX = theta[6]  # maximum soil moisture storage first soil layer [mm]
        SS2MAX = theta[12]  # maximum soil moisture storage second soil layer [mm]
        D = theta[15]  # Flow delay before surface runoff [d]

        # initial store values
        self.S0 = 0.8 * np.array([SIMAX, SMPMAX, SS1MAX, SS2MAX])

        # Added to reinitialize the self.results array
        # Not necessary in MatLab, but is necessary in python to store results
        self.fluxes = np.zeros((self.input_climate['precip'].shape[0], self.num_fluxes))
        self.stores = np.zeros((self.input_climate['precip'].shape[0], self.num_stores))
        self.solver_data = {
            'resnorm': np.zeros(self.input_climate['precip'].shape[0]),
            'solver': np.full(self.input_climate['precip'].shape[0], 'None'),
            'iter': np.zeros(self.input_climate['precip'].shape[0])}

        # initialise the unit hydrographs and still-to-flow vectors
        uh_q0r = uh_4_full(D, delta_t)

        self.uhs = [uh_q0r]

    def model_fun(self, S):
        """
        Model governing equations in state-space formulation.
        """
        theta = self.theta
        SIMAX = theta[0]    # maximum interception storage [mm]
        A = theta[1]        # splitting coefficient for excess precipitation [-]
        FF = theta[2]       # forest fraction [-]
        SMPMAX = theta[3]   # maximum storage macropores [mm]
        CQMP = theta[4]     # runoff time parameter (fast/slow runoff) first soil layer [1/d]
        XQMP = theta[5]     # runoff scale parameter first soil layer [-]
        SS1MAX = theta[6]   # maximum soil moisture storage first soil layer [mm]
        FCCS1 = theta[7]    # field capacity coefficient fist soil layer [-]
        CFS1 = theta[8]     # maximum infiltration rate first soil layer [-]
        XFS1 = theta[9]     # infiltration loss exponent first soil layer [-]
        CQS1 = theta[10]    # runoff time parameter for (fast/slow runoff) first soil layer [1/d]
        XQS1 = theta[11]    # runoff scale parameter first soil layer [-]
        SS2MAX = theta[12]  # maximum soil moisture storage second soil layer [mm]
        CQS2 = theta[13]    # runoff time parameter for (fast/slow runoff) second soil layer [1/d]
        XQS2 = theta[14]    # runoff scale parameter second soil layer [-]

        # delta_t
        delta_t = self.delta_t

        # unit hydrographs and still-to-flow vectors
        uh_q0r = self.uhs[0]

        # stores
        S1, S2, S3, S4 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        # fluxes functions
        flux_ei = evap_1(S1, Ep, delta_t)
        flux_pex = interception_1(P, S1, SIMAX)
        flux_pexmp = split_1(A, flux_pex)
        flux_pexs1 = split_2(A, flux_pex)
        flux_fmp = infiltration_3(flux_pexmp, S2, SMPMAX)
        flux_qexmp = flux_pexmp - flux_fmp
        flux_qmp = interflow_3(CQMP, XQMP, S2, delta_t)
        flux_pqexs1 = flux_pexs1 + flux_qexmp
        flux_fs1 = infiltration_7(CFS1, XFS1, S3, SS1MAX, flux_pqexs1)
        flux_etas1 = evap_23(FF, FCCS1, S3, SS1MAX, Ep, delta_t)
        flux_qs1 = interflow_9(S3, CQS1, FCCS1 * SS1MAX, XQS1, delta_t)
        flux_q0 = flux_pqexs1 - flux_fs1
        flux_q0r = route(flux_q0, uh_q0r)
        flux_qmps1 = flux_qmp + flux_qs1
        flux_pc = infiltration_3(flux_qmps1, S4, SS2MAX)
        flux_qh = flux_qmps1 - flux_pc
        flux_qs2 = interflow_3(CQS2, XQS2, S4, delta_t)
        flux_qgw = 0.0195

        # stores ODEs
        dS1 = P - flux_ei - flux_pex
        dS2 = flux_fmp - flux_qmp
        dS3 = flux_fs1 - flux_etas1 - flux_qs1
        dS4 = flux_pc - flux_qs2

        # outputs
        dS = np.array([dS1, dS2, dS3, dS4])
        fluxes = np.array([flux_ei, flux_pex, flux_pexmp, flux_pexs1,
                           flux_fmp, flux_qexmp, flux_qmp, flux_pqexs1,
                           flux_fs1, flux_etas1, flux_qs1, flux_q0,
                           flux_q0r, flux_qmps1, flux_pc, flux_qh,
                           flux_qs2, flux_qgw])

        return dS, fluxes

    def step(self):
        """
        Runs at the end of every timestep.
        """
        # unit hydrographs and still-to-flow vectors
        uh_q0r = self.uhs[0]

        # input fluxes to the unit hydrographs
        fluxes = self.fluxes[self.t, :]
        flux_q0 = fluxes[11]

        # update still-to-flow vectors using fluxes at current step and
        # unit hydrographs
        uh_q0r = update_uh(uh_q0r, flux_q0)
        self.uhs = [uh_q0r]
