from spotpy.parameter import Uniform
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import rmse

from pymarrmot.models.models.m_37_hbv_15p_5s import m_37_hbv_15p_5s

import pandas as pd
import numpy as np

from pathlib import Path

class spotpy_setup(object):

    # Step 1: Define the parameters of the model as class parameters
    par_ranges = {
        'tt' : [-3, 5],        # TT, threshold temperature for snowfall [oC]
        'tti' : [0, 17],       # TTI, interval length of rain-snow spectrum [oC]
        'ttm' : [-3, 3],       # TTM, threshold temperature for snowmelt [oC]
        'cfr' : [0, 1],        # CFR, coefficient of refreezing of melted snow [-]
        'cfmax' : [0, 20],     # CFMAX, degree-day factor of snowmelt and refreezing [mm/oC/d]
        'whc' : [0, 1],        # WHC, maximum water holding content of snow pack [-]
        'cflux' : [0, 4],      # CFLUX, maximum rate of capillary rise [mm/d]
        'fc' : [1, 2000],      # FC, maximum soil moisture storage [mm]
        'lp' : [0.05, 0.95],   # LP, wilting point as fraction of FC [-]
        'beta' : [0, 10],      # BETA, non-linearity coefficient of upper zone recharge [-]
        'k0' : [0, 1],         # K0, runoff coefficient from upper zone [d-1]
        'alpha' : [0, 4],      # ALPHA, non-linearity coefficient of runoff from upper zone [-]
        'perc': [0, 20],       # PERC, maximum rate of percolation to lower zone [mm/d]
        'k1' : [0, 1],         # K1, runoff coefficient from lower zone [d-1]
        'maxbas' : [1, 120]    # MAXBAS, flow routing delay [d]
    }
    
    # parameters that will be calibrated by Spotpy
    tt = Uniform(low=par_ranges['tt'][0], high=par_ranges['tt'][1], optguess=0)
    tti = Uniform(low=par_ranges['tti'][0], high=par_ranges['tti'][1], optguess=8)
    ttm = Uniform(low=par_ranges['ttm'][0], high=par_ranges['ttm'][1], optguess=0)
    cfr = Uniform(low=par_ranges['cfr'][0], high=par_ranges['cfr'][1], optguess=0.5)
    cfmax = Uniform(low=par_ranges['cfmax'][0], high=par_ranges['cfmax'][1], optguess=10)
    whc = Uniform(low=par_ranges['whc'][0], high=par_ranges['whc'][1], optguess=0.5)
    cflux = Uniform(low=par_ranges['cflux'][0], high=par_ranges['cflux'][1], optguess=2)
    fc = Uniform(low=par_ranges['fc'][0], high=par_ranges['fc'][1], optguess=1000)
    lp = Uniform(low=par_ranges['lp'][0], high=par_ranges['lp'][1], optguess=0.5)
    beta = Uniform(low=par_ranges['beta'][0], high=par_ranges['beta'][1], optguess=5)
    k0 = Uniform(low=par_ranges['k0'][0], high=par_ranges['k0'][1], optguess=0.5)
    alpha = Uniform(low=par_ranges['alpha'][0], high=par_ranges['alpha'][1], optguess=2)
    perc = Uniform(low=par_ranges['perc'][0], high=par_ranges['perc'][1], optguess=10)
    k1 = Uniform(low=par_ranges['k1'][0], high=par_ranges['k1'][1], optguess=0.5)
    maxbas = Uniform(low=par_ranges['maxbas'][0], high=par_ranges['maxbas'][1], optguess=100)

    # initialize the model class m
    m = m_37_hbv_15p_5s()

    # Step 2: Write the def init function, which takes care of any things which need to be done only once
    def __init__(self,obj_func=None):
        self.obj_func = obj_func
            
    # Step 3: Write the def simulation function, which starts your model and returns the results
    def simulation(self,x):
        #update theta with the new parameter set
        self.m.theta = np.array(x)
        
        #execute the model
        (output_ex, output_in, output_ss, output_waterbalance) = self.m.get_output(nargout=4)

        # some of the models output Q as a 2-D array (with only one column)
        if len(output_ex['Q'].shape) == 2:
            q_out = output_ex['Q'][:,0]
        else:
            q_out = output_ex['Q']
        
        return q_out
    
    # Step 4: Write the def evaluation function, which returns the observations
    def evaluation(self):
        return self.trueObs
    
    # Step 5: Write the def objectivefunction function, which returns the objective function value
    def objectivefunction(self, simulation, evaluation, params=None):

        mask = ~np.isnan(evaluation)

        simulation_filtered = simulation[mask]
        evaluation_filtered = evaluation[mask]

        if not self.obj_func:
            # This is used if not overwritten by user
            score = rmse(evaluation_filtered, simulation_filtered)
            like = score

        elif self.obj_func == 'kge_q90':

            q90 = np.nanpercentile(evaluation_filtered, 90)
            q90_mask = evaluation_filtered > q90

            evaluation_q90 = evaluation_filtered[q90_mask]
            simulation_q90 = simulation_filtered[q90_mask]

            score = kge(evaluation_q90, simulation_q90)
            like = -1*score            
            
        else:
            # Spotpy minimizes the objective function, so for objective functions where fitness improves with increasing result, we need to multiply by -1
            score = self.obj_func(evaluation_list, simulation_list)
            like = -1*score

        return like