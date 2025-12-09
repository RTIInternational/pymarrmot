import numpy as np
from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.evaporation.evap_7 import evap_7
from pymarrmot.models.flux.evaporation.evap_3 import evap_3
from pymarrmot.models.flux.saturation.saturation_1 import saturation_1
from pymarrmot.models.flux.interflow.interflow_8 import interflow_8

class m_03_collie2_4p_1s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Collie River v2

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model reference:
    Jothityangkoon, C., M. Sivapalan, and D. Farmer (2001), 'Process controls
    of water balance variability in a large semi-arid catchment: downward 
    approach to hydrological model development.' Journal of Hydrology, 254,
    174-198. doi: 10.1016/S0022-1694(01)00496-6.
    """

    def __init__(self):
        """
        Initialize the Collie River v2 model.
        """
        super().__init__()
        self.num_stores = 1
        self.num_fluxes = 4
        self.num_params = 4

        self.jacob_pattern = [1]

        self.par_ranges = [
            [1, 2000],     # Smax [mm]
            [0.05, 0.95],  # fc as fraction of Smax
            [0, 1],        # a, subsurface runoff coefficient [d-1]
            [0.05, 0.95]   # M, fraction forest cover [-]
        ]

        self.store_names = ["S1"]
        self.flux_names = ["eb", "ev", "qse", "qss"]

        self.flux_groups = {
            'Ea': [1, 2],  # Index or indices of fluxes to add to Actual ET
            'Q': [3, 4]    # Index or indices of fluxes to add to Streamflow
        }

    def init(self):
        """
        Initialize the model.
        """
        pass

    def model_fun(self, S):
        """
        Calculate model dynamics.

        Parameters:
        S (array_like): State variables.

        Returns:
        dS (array_like): Derivative of state variables.
        fluxes (array_like): Model fluxes.
        """
        theta = self.theta
        S1max = theta[0]  # Maximum soil moisture storage [mm]
        Sfc = theta[1]    # Field capacity as fraction of S1max [-]
        a = theta[2]      # Subsurface runoff coefficient [d-1]
        M = theta[3]      # Fraction forest cover [-]

        delta_t = self.delta_t

        S1 = S[0]

        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]

        flux_eb = evap_7(S1, S1max, (1 - M) * Ep, delta_t)
        flux_ev = evap_3(Sfc, S1, S1max, M * Ep, delta_t)
        flux_qse = saturation_1(P, S1, S1max)
        flux_qss = interflow_8(S1, a, Sfc * S1max)

        dS1 = P - flux_eb - flux_ev - flux_qse - flux_qss

        dS = np.array([dS1])
        fluxes = [flux_eb, flux_ev, flux_qse, flux_qss]

        return dS, fluxes

    def step(self):
        """
        Perform model step.
        """
        pass