from spotpy.parameter import Uniform
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import rmse

from pymarrmot.models.models.m_27_tank_12p_4s import m_27_tank_12p_4s

import pandas as pd
import numpy as np

from pathlib import Path

class spotpy_setup(object):
    
    # Step 1: Define the parameters of the model as class parameters
    par_ranges = {
        'a0' : [0, 1],            # a0, Time parameter for drainage 1>2 [d-1]
        'b0' : [0, 1],            # b0, Time parameter for drainage 2>3 [d-1]
        'c0' : [0, 1],            # c0, Time parameter for drainage 3>4 [d-1]
        'a1' : [0, 1],            # a1, Time parameter for surface runoff 1 [d-1]
        'fa' : [0, 1],            # fa, Fraction of a1 that is a2 [-]
        'fb' : [0, 1],            # fb, Fraction of a2 that is b1 [-]
        'fc' : [0, 1],            # fc, Fraction of b1 that is c1 [-]
        'fd' : [0, 1],            # fd, Fraction of c1 that is d1 [-]
        'st' : [1, 2000],         # st, Maximum soil depth (sum of runoff thresholds) [mm]
        'f2' : [0.01, 0.99],      # f2, Fraction of st that consitutes threshold t2 [-]
        'f1' : [0.01, 0.99],      # f1, Fraction of st-t2 that is added to t2 to find threshold 1 [-] (ensures t1 > t2)
        'f3' : [0.01, 0.99]       # f3, Fraction of st-t1-t2 that consitutes threshold 3 [-]
    }

    # parameters that will be calibrated by Spotpy
    a0 = Uniform(low=par_ranges['a0'][0], high=par_ranges['a0'][1], optguess=0.5)
    b0 = Uniform(low=par_ranges['b0'][0], high=par_ranges['b0'][1], optguess=0.5)
    c0 = Uniform(low=par_ranges['c0'][0], high=par_ranges['c0'][1], optguess=0.5)
    a1 = Uniform(low=par_ranges['a1'][0], high=par_ranges['a1'][1], optguess=0.5)
    fa = Uniform(low=par_ranges['fa'][0], high=par_ranges['fa'][1], optguess=0.5)
    fb = Uniform(low=par_ranges['fb'][0], high=par_ranges['fb'][1], optguess=0.5)
    fc = Uniform(low=par_ranges['fc'][0], high=par_ranges['fc'][1], optguess=0.5)
    fd = Uniform(low=par_ranges['fd'][0], high=par_ranges['fd'][1], optguess=0.5)
    st = Uniform(low=par_ranges['st'][0], high=par_ranges['st'][1], optguess=1000)
    f2 = Uniform(low=par_ranges['f2'][0], high=par_ranges['f2'][1], optguess=0.5)
    f1 = Uniform(low=par_ranges['f1'][0], high=par_ranges['f1'][1], optguess=0.5)
    f3 = Uniform(low=par_ranges['f3'][0], high=par_ranges['f3'][1], optguess=0.5)


    # initialize the model class m
    m = m_27_tank_12p_4s()

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