import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.saturation import saturation_3
from pymarrmot.models.flux.evaporation import evap_1, evap_3
from pymarrmot.models.flux.percolation import percolation_2
from pymarrmot.models.flux.split import split_1
from pymarrmot.models.flux.baseflow import baseflow_1
from pymarrmot.models.flux.snowfall import snowfall_1
from pymarrmot.models.flux.rainfall import rainfall_1
from pymarrmot.models.flux.melt import melt_1
from pymarrmot.models.flux.interception import interception_1
from pymarrmot.models.unit_hydro import (uh_3_half, route, update_uh)

class m_34_flexis_12p_5s(MARRMoT_model):
    def __init__(self):
        super().__init__()
        self.num_stores = 5                                             # number of model stores
        self.num_fluxes = 14                                            # number of model fluxes
        self.num_params = 12
            
        self.jacob_pattern = np.array([[1, 0, 0, 0, 0],
                                      [1, 1, 0, 0, 0],
                                      [1, 1, 1, 0, 0],
                                      [1, 1, 1, 1, 0],
                                      [1, 1, 1, 0, 1]])              # Jacobian matrix of model store ODEs
                             
        self.par_ranges = np.array([[1, 2000],       # URmax, Maximum soil moisture storage [mm]
                                    [0, 10],         # beta, Unsaturated zone shape parameter [-]
                                    [0, 1],          # D, Fast/slow runoff distribution parameter [-]
                                    [0, 20],         # PERCmax, Maximum percolation rate [mm/d]
                                    [0.05, 0.95],    # Lp, Wilting point as fraction of s1max [-]
                                    [1, 5],          # Nlagf, Flow delay before fast runoff [d]
                                    [1, 15],         # Nlags, Flow delay before slow runoff [d]
                                    [0, 1],          # Kf, Fast runoff coefficient [d-1]
                                    [0, 1],          # Ks, Slow runoff coefficient [d-1]
                                    [0, 5],          # Imax, Maximum interception storage [mm]
                                    [-3, 5],         # TT, Threshold temperature for snowfall/snowmelt [oC]
                                    [0, 20]])        # ddf, Degree-day factor for snowmelt [mm/d/oC]
            
        self.store_names = ["S1", "S2", "S3", "S4", "S5"]                   # Names for the stores
        self.flux_names  = ["ps", "pi", "m", "peff", "ei",
                           "ru", "eur", "rp", "rf", "rs",
                           "rf1", "rs1", "qf", "qs"]                     # Names for the fluxes
            
        self.flux_groups = {'Ea': [5, 7],                                   # Index or indices of fluxes to add to Actual ET
                           'Q': [13, 14]}                                  # Index or indices of fluxes to add to Streamflow
            
    def init(self):
        # parameters
        theta = self.theta
        delta_t = self.delta_t
        nlagf = theta[5]     # Flow delay before fast runoff [d]
        nlags = theta[6]     # Flow delay before slow runoff [d]
            
        # initialise the unit hydrographs and still-to-flow vectors
        uh_f = uh_3_half(nlagf, delta_t)
        uh_s = uh_3_half(nlags, delta_t)
            
        self.uhs = [uh_f, uh_s]
        
    def model_fun(self, S):
        # parameters
        theta = self.theta
        smax = theta[0]     # Maximum soil moisture storage [mm]
        beta = theta[1]     # Unsaturated zone shape parameter [-]
        d = theta[2]        # Fast/slow runoff distribution parameter [-]
        percmax = theta[3]  # Maximum percolation rate [mm/d]
        lp = theta[4]       # Wilting point as fraction of s1max [-]
        nlagf = theta[5]    # Flow delay before fast runoff [d]
        nlags = theta[6]    # Flow delay before slow runoff [d]
        kf = theta[7]       # Fast runoff coefficient [d-1]
        ks = theta[8]       # Slow runoff coefficient [d-1]
        imax = theta[9]     # Maximum interception storage [mm]
        tt = theta[10]      # Threshold temperature for snowfall/snowmelt [oC]
        ddf = theta[11]     # Degree-day factor for snowmelt [mm/d/oC]
            
        # delta_t
        delta_t = self.delta_t
            
        # unit hydrographs and still-to-flow vectors
        uhs = self.uhs
        uh_f = uhs[0]
        uh_s = uhs[1]
            
        # stores
        S1 = S[0]
        S2 = S[1]
        S3 = S[2]
        S4 = S[3]
        S5 = S[4]
            
        # climate input at time t
        t = self.t
        P = self.input_climate['precip'][t]
        Ep = self.input_climate['pet'][t]
        T = self.input_climate['temp'][t]
            
        # fluxes functions
        flux_ps = snowfall_1(P, T, tt)
        flux_pi = rainfall_1(P, T, tt)
        flux_m = melt_1(ddf, tt, T, S1, delta_t)
        flux_peff = interception_1(flux_m + flux_pi, S2, imax)
        flux_ei = evap_1(S2, Ep, delta_t)
        flux_ru = saturation_3(S3, smax, beta, flux_peff)
        flux_eur = evap_3(lp, S3, smax, Ep, delta_t)
        flux_rp = percolation_2(percmax, S3, smax, delta_t)
        flux_rf = split_1(1 - d, flux_peff - flux_ru)
        flux_rs = split_1(d, flux_peff - flux_ru)
        flux_rfl = route(flux_rf, uh_f)
        flux_rsl = route(flux_rs + flux_rp, uh_s)
        flux_qf = baseflow_1(kf, S4)
        flux_qs = baseflow_1(ks, S5)
            
        # stores ODEs
        dS1 = flux_ps - flux_m
        dS2 = flux_m + flux_pi - flux_peff - flux_ei
        dS3 = flux_ru - flux_eur - flux_rp
        dS4 = flux_rfl - flux_qf
        dS5 = flux_rsl - flux_qs 

        # outputs
        dS = np.array([dS1, dS2, dS3, dS4, dS5])
        fluxes = np.array([flux_ps, flux_pi, flux_m, flux_peff, flux_ei,
                           flux_ru, flux_eur, flux_rp, flux_rf, flux_rs, 
                           flux_rfl, flux_rsl, flux_qf, flux_qs])

        return dS, fluxes

    def step(self):
        # unit hydrographs and still-to-flow vectors
        uhs = self.uhs
        uh_f = uhs[0]
        uh_s = uhs[1]
            
        # input fluxes to the unit hydrographs
        fluxes = self.fluxes[self.t, :]
        flux_rp = fluxes[7]
        flux_rf = fluxes[8]
        flux_rs = fluxes[9]
            
        # update still-to-flow vectors using fluxes at current step and
        # unit hydrographs
        uh_f = update_uh(uh_f, flux_rf)
        uh_s = update_uh(uh_s, flux_rs + flux_rp)
            
        self.uhs = [uh_f, uh_s]
