import numpy as np

from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux.split import split_1
from pymarrmot.models.flux.soil_moisture import soilmoisture_1
from pymarrmot.models.flux.evaporation import evap_7
from pymarrmot.models.flux.saturation import saturation_1
from pymarrmot.models.flux.interflow import interflow_5
from pymarrmot.models.flux.evaporation import evap_1
from pymarrmot.models.flux.percolation import percolation_4
from pymarrmot.models.flux.soil_moisture import soilmoisture_2
from pymarrmot.models.flux.baseflow import baseflow_1
from pymarrmot.models.auxiliary.deficit_based_distribution import deficit_based_distribution

class m_33_sacramento_11p_5s(MARRMoT_model):
    """Class for hydrologic conceptual model: Sacramento-SMA

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model references
    National Weather Service (2005), II.3-SAC-SMA: Conceptualization of the 
    Sacramento Soil Moisture Accounting model. In National Weather Service 
    River Forecast System (NWSRFS) User Manual, 1-13
    
    Koren, V. I., Smith, M., Wang, D., & Zhang, Z. (2000). Use of soil 
    property data in the derivation of conceptual rainfall-runoff model 
    parameters. Proceedings of the 15th Conference on Hydrology, AMS, Long 
    Beach, CA, (1), 103�106."""
    
    def __init__(self):
        super().__init__()
        self.theta_derived = None

        self.num_stores = 5
        self.num_fluxes = 20
        self.num_params = 11
        self.jacob_pattern = np.array([[1,1,0,0,0],
                                      [1,1,1,1,1],
                                      [1,1,1,1,1],
                                      [0,1,1,1,1],
                                      [0,1,1,1,1]])
        self.par_ranges = np.array([[0       , 1],       # pctim, Fraction impervious area [-]
                                   [1       , 2000],     # smax, Maximum total storage depth [mm]
                                   [0.005   , 0.995],    # f1, fraction of smax that is Maximum upper zone tension water storage [mm]
                                   [0.005   , 0.995],    # f2, fraction of smax-S1max that is Maximum upper zone free water storage [mm]
                                   [0       , 1],        # kuz, Interflow runoff coefficient [d-1]
                                   [0       , 7],        # rexp, Base percolation rate non-linearity factor [-]
                                   [0.005   , 0.995],    # f3, fraction of smax-S1max-S2max that is  Maximum lower zone tension water storage [mm]
                                   [0.005   , 0.995],    # f4, fraction of smax-S1max-S2max-S3max that is  Maximum lower zone primary free water storage [mm]
                                   [0       , 1],        # pfree, Fraction of percolation directed to free water stores [-]
                                   [0       , 1],        # klzp, Primary baseflow runoff coefficient [d-1]
                                   [0       , 1]])       # klzs, Supplemental baseflow runoff coefficient [d-1]
        self.store_names = ["S1", "S2", "S3", "S4", "S5"]
        self.flux_names  = ["qdir", "peff", "ru", "euztw", "twexu",
                           "qsur", "qint", "euzfw", "pc", "pctw",
                           "elztw", "twexl", "twexlp", "twexls", "pcfwp",
                           "pcfws", "rlp", "rls", "qbfp", "qbfs"]
        self.flux_groups = {"Ea": [4, 8, 11], "Q": [1, 6, 7, 19, 20]}

    def init(self):
        theta = self.theta
        smax    = theta[1]     # Maximum total storage depth [mm]
        f1      = theta[2]     # fraction of smax that is Maximum upper zone tension water storage (uztwm) [-]
        f2      = theta[3]     # fraction of smax-uztwm that is Maximum upper zone free water storage (uzfwm) [-]
        f3      = theta[6]     # fraction of smax-uztwm-uzfwm that is Maximum lower zone tension water storage (lztwm) [-]
        f4      = theta[7]     # fraction of smax-uztwm-uzfwm-lztwm that is Maximum lower zone primary free water storage (lzfwpm) [-]
        klzp    = theta[9]     # Primary baseflow runoff coefficient [d-1]
        klzs    = theta[10]    # Supplemental baseflow runoff coefficient [d-1]
        uztwm   = f1*smax                                             # Maximum upper zone tension water storage [mm]
        uzfwm   = max(0.005/4,f2*(smax-uztwm))                        # Maximum upper zone free water storage [mm]
        lztwm   = max(0.005/4,f3*(smax-uztwm-uzfwm))                  # Maximum lower zone tension water storage [mm]
        lzfwpm  = max(0.005/4,f4*(smax-uztwm-uzfwm-lztwm))            # Maximum lower zone primary free water storage [mm]
        lzfwsm  = max(0.005/4,(1-f4)*(smax-uztwm-uzfwm-lztwm))        # Maximum lower zone supplemental free water storage [mm]
        pbase   = lzfwpm*klzp + lzfwsm*klzs                           # Base percolation rate [mm/d]
        zperc   = min(100000,
                      (lztwm+lzfwsm*(1-klzs))/(lzfwsm*klzs+lzfwpm*klzp) + \
                      (lzfwpm*(1-klzp))/(lzfwsm*klzs+lzfwpm*klzp))    # Base percolation rate multiplication factor [-]: can return Inf, hence the min(10000,...)
        self.theta_derived = [uztwm, uzfwm, lztwm, lzfwpm, lzfwsm,
                             pbase, zperc]
        self.store_max = np.array([uztwm,uzfwm,lztwm,lzfwpm,lzfwsm])

    def model_fun(self, S: list[float]) -> tuple[list[float], list[float]]:
        theta   = self.theta
        pctim   = theta[0]     # Fraction impervious area [-]
        kuz     = theta[4]     # Interflow runoff coefficient [d-1]
        rexp    = theta[5]     # Base percolation rate non-linearity factor [-]
        pfree   = theta[8]     # Fraction of percolation directed to free water stores [-]
        klzp    = theta[9]     # Primary baseflow runoff coefficient [d-1]
        klzs    = theta[10]    # Supplemental baseflow runoff coefficient [d-1]
        theta_d = self.theta_derived
        uztwm   = theta_d[0]   # Maximum upper zone tension water storage [mm]
        uzfwm   = theta_d[1]   # Maximum upper zone free water storage [mm]
        lztwm   = theta_d[2]   # Maximum lower zone tension water storage [mm]
        lzfwpm  = theta_d[3]   # Maximum lower zone primary free water storage [mm]
        lzfwsm  = theta_d[4]   # Maximum lower zone supplemental free water storage [mm]
        pbase   = theta_d[5]   # Base percolation rate [mm/d]
        zperc   = theta_d[6]   # Base percolation rate multiplication factor [-]
        delta_t = self.delta_t

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

        flux_qdir    = split_1(pctim,P)
        flux_peff    = split_1(1-pctim,P)
        flux_ru      = soilmoisture_1(S1,uztwm,S2,uzfwm)
        flux_euztw   = evap_7(S1,uztwm,Ep,delta_t)
        flux_twexu   = saturation_1(flux_peff,S1,uztwm)
        flux_qsur    = saturation_1(flux_twexu,S2,uzfwm)
        flux_qint    = interflow_5(kuz,S2)
        flux_euzfw   = evap_1(S2,max(0,Ep-flux_euztw),delta_t)
        flux_pc      = percolation_4(pbase,zperc,rexp,max(0,lztwm-S3)+max(0,lzfwpm-S4)+max(0,lzfwsm-S5),lztwm+lzfwpm+lzfwsm,S2,uzfwm,delta_t)
        flux_pctw    = split_1(1-pfree,flux_pc)
        flux_elztw   = evap_7(S3,lztwm,max(0,Ep-flux_euztw-flux_euzfw),delta_t)
        flux_twexl   = saturation_1(flux_pctw,S3,lztwm)

        # modified from MatLab to specify the first position in the tuple returned from deficit_based_distribution
        # Matlab will use only the first of the two outputs by default, but Python requires us to specify this
        flux_twexlp  = split_1(deficit_based_distribution(S4,lzfwpm,S5,lzfwsm)[0],flux_twexl)
        flux_twexls  = split_1(deficit_based_distribution(S5,lzfwsm,S4,lzfwpm)[0],flux_twexl)
        flux_pcfwp   = split_1(pfree*deficit_based_distribution(S4,lzfwpm,S5,lzfwsm)[0],flux_pc)
        flux_pcfws   = split_1(pfree*deficit_based_distribution(S5,lzfwsm,S4,lzfwpm)[0],flux_pc)

        flux_rlp     = soilmoisture_2(S3,lztwm,S4,lzfwpm,S5,lzfwsm)
        flux_rls     = soilmoisture_2(S3,lztwm,S5,lzfwsm,S4,lzfwpm)   
        flux_qbfp    = baseflow_1(klzp,S4)
        flux_qbfs    = baseflow_1(klzs,S5)

        dS1 = flux_peff   + flux_ru    - flux_euztw - flux_twexu
        dS2 = flux_twexu  - flux_euzfw - flux_qsur  - flux_qint  - flux_ru - flux_pc
        dS3 = flux_pctw   + flux_rlp   + flux_rls   - flux_elztw - flux_twexl
        dS4 = flux_twexlp + flux_pcfwp - flux_rlp   - flux_qbfp
        dS5 = flux_twexls + flux_pcfws - flux_rls   - flux_qbfs

        dS = np.array([dS1, dS2, dS3, dS4, dS5])
        fluxes = [flux_qdir, flux_peff, flux_ru, flux_euztw, flux_twexu,
                  flux_qsur, flux_qint, flux_euzfw, flux_pc, flux_pctw,
                  flux_elztw, flux_twexl, flux_twexlp, flux_twexls, flux_pcfwp,
                  flux_pcfws, flux_rlp, flux_rls, flux_qbfp, flux_qbfs]

        return dS, fluxes

    def step(self):
        pass