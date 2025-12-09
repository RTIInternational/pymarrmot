from spotpy.parameter import Uniform
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import rmse

from pymarrmot.models.models.m_05_ihacres_7p_1s import m_05_ihacres_7p_1s

import pandas as pd
import numpy as np

from pathlib import Path

class spotpy_setup(object):

    # Step 1: Define the parameters of the model as class parameters
    par_ranges = {
        'lp' : [1, 2000],    # lp, Wilting point [mm]
        'd' :  [1, 2000],    # d, Threshold for flow generation [mm]
        'p' :  [0, 10],      # p, Flow response non-linearity [-]
        'alpha' : [0, 1],       # alpha, Fast/slow flow division [-]
        'tau_q' : [1, 700],     # tau_q, Fast flow routing delay [d]
        'tau_s' : [1, 700],     # tau_s, Slow flow routing delay [d]
        'tau_d' : [0, 119]      # tau_d, flow delay [d]
    }
    
    # parameters that will be calibrated by Spotpy
    lp = Uniform(low=par_ranges['lp'][0], high=par_ranges['lp'][1], optguess=1000)
    d = Uniform(low=par_ranges['d'][0], high=par_ranges['d'][1], optguess=1000)
    p = Uniform(low=par_ranges['p'][0], high=par_ranges['p'][1], optguess=5)
    alpha = Uniform(low=par_ranges['alpha'][0], high=par_ranges['alpha'][1], optguess=0.5)
    tau_q = Uniform(low=par_ranges['tau_q'][0], high=par_ranges['tau_q'][1], optguess=400)
    tau_s = Uniform(low=par_ranges['tau_s'][0], high=par_ranges['tau_s'][1], optguess=400)
    tau_d = Uniform(low=par_ranges['tau_d'][0], high=par_ranges['tau_d'][1], optguess=50)

    # initialize the model class m
    m = m_05_ihacres_7p_1s()

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

            # minimize kge for top 90th percentile flows only
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