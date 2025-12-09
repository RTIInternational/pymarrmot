from spotpy.parameter import Uniform
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import rmse

from pymarrmot.models.models.m_40_smar_8p_6s import m_40_smar_8p_6s

import pandas as pd
import numpy as np

from pathlib import Path

class spotpy_setup(object):
    
    # Step 1: Define the parameters of the model as class parameters
    par_ranges = {
        'h': [0, 1],             # h, Maximum fraction of direct runoff [-]
        'y': [0, 200],           # y, Infiltration rate [mm/d]
        'smax': [1, 2000],       # smax, Maximum soil moisture storage [mm]
        'c': [0, 1],             # c, Evaporation reduction coefficient [-]
        'g': [0, 1],             # g, Groundwater recharge coefficient [-]
        'kg': [0, 1],            # kg, Groundwater time parameter [d-1]
        'n': [1, 10],            # n, Number of Nash cascade reservoirs [-]
        'nk': [1, 120]           # n*k, Routing delay [d]
    }

    # parameters that will be calibrated by Spotpy
    h = Uniform(low=par_ranges['h'][0], high=par_ranges['h'][1], optguess=0.5)
    y = Uniform(low=par_ranges['h'][0], high=par_ranges['h'][1], optguess=100)
    smax = Uniform(low=par_ranges['h'][0], high=par_ranges['h'][1], optguess=1000)
    c = Uniform(low=par_ranges['h'][0], high=par_ranges['h'][1], optguess=0.5)
    g = Uniform(low=par_ranges['h'][0], high=par_ranges['h'][1], optguess=0.5)
    kg = Uniform(low=par_ranges['h'][0], high=par_ranges['h'][1], optguess=0.5)
    n = Uniform(low=par_ranges['h'][0], high=par_ranges['h'][1], optguess=5)
    nk = Uniform(low=par_ranges['h'][0], high=par_ranges['h'][1], optguess=60)

    # initialize the model class m
    m = m_40_smar_8p_6s()

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