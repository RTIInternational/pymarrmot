import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation import evap_1, evap_6, evap_5
from pymarrmot.models.flux.interception import interception_1
from pymarrmot.models.flux.saturation import saturation_1
from pymarrmot.models.flux.baseflow import baseflow_1
from pymarrmot.models.flux.interflow import interflow_9
from pymarrmot.models.unit_hydro import (route, uh_4_full, update_uh)

class m_16_newzealand2_8p_2s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: New Zealand model v2

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model reference
    Atkinson, S. E., Sivapalan, M., Woods, R. A., & Viney, N. R. (2003).
    Dominant physical controls on hourly flow predictions and the role of
    spatial variability: Mahurangi catchment, New Zealand. Advances in Water
    Resources, 26(3), 219–235. http://doi.org/10.1016/S0309-1708(02)00183-5
    """

    def __init__(self):
        """
        creator method
        """
        super().__init__()
        self.num_stores = 2  # number of model stores
        self.num_fluxes = 8  # number of model fluxes
        self.num_params = 8

        self.jacob_pattern = np.array([[1, 0],
                                      [1, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[0, 5],       # Maximum interception storage [mm]
                                   [1, 2000],    # Smax, Maximum soil moisture storage [mm]
                                   [0.05, 0.95],  # sfc, Field capacity as fraction of maximum soil moisture [-]
                                   [0.05, 0.95],  # m, Fraction forest [-]
                                   [0, 1],       # a, Subsurface runoff coefficient [d-1]
                                   [1, 5],       # b, Runoff non-linearity [-]
                                   [0, 1],       # tcbf, Baseflow runoff coefficient [d-1]
                                   [1, 120]])    # Routing time delay [d]

        self.store_names = ["S1", "S2"]  # Names for the stores
        self.flux_names = ["eint", "qtf", "veg", "ebs",
                          "qse",  "qss", "qbf", "qt"]  # Names for the fluxes

        self.flux_groups = {"Ea": [1, 3, 4],  # Index or indices of fluxes to add to Actual ET
                           "Q": 8}            # Index or indices of fluxes to add to Streamflow

    def init(self):
        """
        INITialisation function
        """
        # parameters
        theta = self.theta
        delta_t = self.delta_t

        s1max = theta[0]  # Maximum interception storage [mm]
        s2max = theta[1]  # Maximum soil moisture storage [mm]
        d = theta[7]      # Routing delay [d]

        # maximum store values
        self.store_max = np.array([s1max, s2max])

        # initialise the unit hydrographs and still-to-flow vectors
        uh = uh_4_full(d, delta_t)

        self.uhs = [uh]

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation
        """
        # parameters
        theta = self.theta
        s1max = theta[0]  # Maximum interception storage [mm]
        s2max = theta[1]  # Maximum soil moisture storage [mm]
        sfc = theta[2]    # Field capacity as fraction of maximum soil moisture [-]
        m = theta[3]      # Fraction forest [-]
        a = theta[4]      # Subsurface runoff coefficient [d-1]
        b = theta[5]      # Runoff non-linearity [-]
        tcbf = theta[6]   # Baseflow runoff coefficient [d-1]

        # delta_t
        delta_t = self.delta_t

        # unit hydrographs and still-to-flow vectors
        uhs = self.uhs
        uh = uhs[0]

        # stores
        S1, S2 = S

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]
        
        # fluxes functions
        flux_eint = evap_1(S1, Ep, delta_t)
        flux_qtf = interception_1(P, S1, s1max)
        flux_veg = evap_6(m, sfc, S2, s2max, Ep, delta_t)
        flux_ebs = evap_5(m, S2, s2max, Ep, delta_t)
        flux_qse = saturation_1(flux_qtf, S2, s2max)
        flux_qss = interflow_9(S2, a, sfc * s2max, b, delta_t)
        flux_qbf = baseflow_1(tcbf, S2)
        flux_qt = route(flux_qse + flux_qss + flux_qbf, uh)

        # stores ODEs
        dS1 = P - flux_eint - flux_qtf
        dS2 = flux_qtf - flux_veg - flux_ebs - flux_qse - flux_qss - flux_qbf

        # outputs
        dS = np.array([dS1, dS2])
        fluxes = np.array([flux_eint, flux_qtf, flux_veg, flux_ebs,
                           flux_qse, flux_qss, flux_qbf, flux_qt])

        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep
        """
        # unit hydrographs and still-to-flow vectors
        uhs = self.uhs
        uh = uhs[0]

        # input fluxes to the unit hydrographs
        fluxes = self.fluxes[self.t]
        flux_qse = fluxes[4]
        flux_qss = fluxes[5]
        flux_qbf = fluxes[6]

        # update still-to-flow vectors using fluxes at current step and
        # unit hydrographs
        uh = update_uh(uh, flux_qse + flux_qss + flux_qbf)
        self.uhs = [uh]
