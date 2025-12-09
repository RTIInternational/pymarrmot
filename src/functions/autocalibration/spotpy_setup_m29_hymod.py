from spotpy.parameter import Uniform
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import rmse

from pymarrmot.models.models.m_29_hymod_5p_5s import m_29_hymod_5p_5s

import pandas as pd
import numpy as np

from pathlib import Path

class spotpy_setup(object):

    # Step 1: Define the parameters of the model as class parameters
    par_ranges = {
        'Smax': [1, 2000],   
        'b': [0.01, 10],        
        'a': [0.01, 1],         
        'kf': [0.01, 1],       
        'ks': [0.01, 1]        
        }
    
    # parameters that will be calibrated by Spotpy
    smax = Uniform(low=par_ranges['Smax'][0], high=par_ranges['Smax'][1], optguess=500)    # Smax, Maximum soil moisture storage [mm]
    b = Uniform(low=par_ranges['b'][0], high=par_ranges['b'][1], optguess=0.1725)          # b, Soil depth distribution parameter [-]
    a = Uniform(low=par_ranges['a'][0], high=par_ranges['a'][1], optguess=0.8127)          # a, Runoff distribution fraction [-]
    kf = Uniform(low=par_ranges['kf'][0], high=par_ranges['kf'][1], optguess=0.0404)       # kf, fast flow time parameter [d-1]
    ks = Uniform(low=par_ranges['ks'][0], high=par_ranges['ks'][1], optguess=0.5592)       # ks, base flow time parameter [d-1]   

    # initialize the model class m
    m = m_29_hymod_5p_5s()

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