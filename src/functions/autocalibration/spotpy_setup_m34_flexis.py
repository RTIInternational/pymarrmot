from spotpy.parameter import Uniform
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import rmse

from pymarrmot.models.models.m_34_flexis_12p_5s import m_34_flexis_12p_5s

import pandas as pd
import numpy as np

from pathlib import Path

class spotpy_setup(object):

    # Step 1: Define the parameters of the model as class parameters
    par_ranges = {
        'smax' : [1, 2000],          # URmax, Maximum soil moisture storage [mm]
        'beta' : [0, 10],            # beta, Unsaturated zone shape parameter [-]
        'd' : [0, 1],                # D, Fast/slow runoff distribution parameter [-]
        'percmax' : [0, 20],         # PERCmax, Maximum percolation rate [mm/d]
        'lp' : [0.05, 0.95],         # Lp, Wilting point as fraction of s1max [-]
        'nlagf' : [1, 5],            # Nlagf, Flow delay before fast runoff [d]
        'nlags' : [1, 15],           # Nlags, Flow delay before slow runoff [d]
        'kf' : [0, 1],               # Kf, Fast runoff coefficient [d-1]
        'ks' : [0, 1],               # Ks, Slow runoff coefficient [d-1]
        'imax' : [0, 5],             # Imax, Maximum interception storage [mm]
        'tt' : [-3, 5],              # TT, Threshold temperature for snowfall/snowmelt [oC]
        'ddf' : [0, 20]              # ddf, Degree-day factor for snowmelt [mm/d/oC]
    }
    
    # parameters that will be calibrated by Spotpy
    smax = Uniform(low=par_ranges['smax'][0], high=par_ranges['smax'][1], optguess=1000)
    beta = Uniform(low=par_ranges['beta'][0], high=par_ranges['beta'][1], optguess=5)
    d = Uniform(low=par_ranges['d'][0], high=par_ranges['d'][1], optguess=0.5)
    percmax = Uniform(low=par_ranges['percmax'][0], high=par_ranges['percmax'][1], optguess=10)
    lp = Uniform(low=par_ranges['lp'][0], high=par_ranges['lp'][1], optguess=0.5)
    nlagf = Uniform(low=par_ranges['nlagf'][0], high=par_ranges['nlagf'][1], optguess=3)
    nlags = Uniform(low=par_ranges['nlags'][0], high=par_ranges['nlags'][1], optguess=8)
    kf = Uniform(low=par_ranges['kf'][0], high=par_ranges['kf'][1], optguess=0.5)
    ks = Uniform(low=par_ranges['ks'][0], high=par_ranges['ks'][1], optguess=0.5)
    imax = Uniform(low=par_ranges['imax'][0], high=par_ranges['imax'][1], optguess=3)
    tt = Uniform(low=par_ranges['tt'][0], high=par_ranges['tt'][1], optguess=1)
    ddf = Uniform(low=par_ranges['ddf'][0], high=par_ranges['ddf'][1], optguess=10)

    # initialize the model class m
    m = m_34_flexis_12p_5s()

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