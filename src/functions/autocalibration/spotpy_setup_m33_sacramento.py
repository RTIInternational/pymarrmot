from spotpy.parameter import Uniform
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import rmse

from pymarrmot.models.models.m_33_sacramento_11p_5s import m_33_sacramento_11p_5s

import pandas as pd
import numpy as np

from pathlib import Path

class spotpy_setup(object):

    # Step 1: Define the parameters of the model as class parameters
    par_ranges = {
        'pctim': [0, 0.05],          # pctim, Fraction impervious area [-]
        'uztwm' : [25, 125],
        'uzfwm' : [10, 75], 
        'lztwm' : [75, 300],
        'kuz' : [0.2, 0.5],
        'rexp' : [1.4, 3.5],
        'lzfpm' : [40, 600],
        'lzfsm' : [15, 300],
        'pfree' : [0, 0.5],
        'lzpk' : [0.001, 0.015],
        'lzsk' : [0.03, 0.2],
        'zperc' : [20, 300],
    }

    pctim = Uniform(low=par_ranges['pctim'][0], high=par_ranges['pctim'][1], optguess=0)
    uztwm = Uniform(low=par_ranges['uztwm'][0], high=par_ranges['uztwm'][1], optguess=30)
    uzfwm = Uniform(low=par_ranges['uzfwm'][0], high=par_ranges['uzfwm'][1], optguess=20)
    lztwm = Uniform(low=par_ranges['lztwm'][0], high=par_ranges['lztwm'][1], optguess=80)
    kuz = Uniform(low=par_ranges['kuz'][0], high=par_ranges['kuz'][1], optguess=0.5)
    rexp = Uniform(low=par_ranges['rexp'][0], high=par_ranges['rexp'][1], optguess=3.0)
    lzfpm = Uniform(low=par_ranges['lzfpm'][0], high=par_ranges['lzfpm'][1], optguess=50)
    lzfsm = Uniform(low=par_ranges['lzfsm'][0], high=par_ranges['lzfsm'][1], optguess=30)
    pfree = Uniform(low=par_ranges['pfree'][0], high=par_ranges['pfree'][1], optguess=0.3)
    lzpk = Uniform(low=par_ranges['lzpk'][0], high=par_ranges['lzpk'][1], optguess=0.01)
    lzsk = Uniform(low=par_ranges['lzsk'][0], high=par_ranges['lzsk'][1], optguess=0.1)
    zperc = Uniform(low=par_ranges['zperc'][0], high=par_ranges['zperc'][1], optguess=50)

    # initialize the model class m
    m = m_33_sacramento_11p_5s()

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