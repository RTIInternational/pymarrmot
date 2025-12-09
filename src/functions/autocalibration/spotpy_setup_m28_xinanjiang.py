from spotpy.parameter import Uniform
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import rmse

from pymarrmot.models.models.m_28_xinanjiang_12p_4s import m_28_xinanjiang_12p_4s

import pandas as pd
import numpy as np

from pathlib import Path

class spotpy_setup(object):

    # Step 1: Define the parameters of the model as class parameters
    par_ranges = {
        'aim' : [0, 1],                           # aim,  Fraction impervious area [-]
        'a' : [-0.49, 0.49],                   # a,    Tension water distribution inflection parameter [-]
        'b' : [0, 10],                         # b,    Tension water distribution shape parameter [-]
        'stot' : [1, 2000],                       # stot, Total soil moisture storage (W+S) [mm]
        'fwm' : [0.01, 0.99],                    # fwm,  Fraction of Stot that is Wmax [-]
        'flm' : [0.01, 0.99],                    # flm,  Fraction of wmax that is LM [-]
        'c' : [0.01, 0.99],                    # c,    Fraction of LM for second evaporation change [-]
        'ex' : [0, 10],                         # ex,   Free water distribution shape parameter [-]
        'ki' : [0, 1],                          # ki,   Free water interflow parameter [d-1]
        'kg' : [0, 1],                          # kg,   Free water groundwater parameter [d-1]
        'ci' : [0, 1],                          # ci,   Interflow time coefficient [d-1]
        'cg' : [0, 1]                           # cg,   Baseflow time coefficient [d-1]
    }    
    
    # parameters that will be calibrated by Spotpy
    aim = Uniform(low=par_ranges['aim'][0], high=par_ranges['aim'][1], optguess=0.5)
    a = Uniform(low=par_ranges['a'][0], high=par_ranges['a'][1], optguess=0)
    b = Uniform(low=par_ranges['b'][0], high=par_ranges['b'][1], optguess=5)
    stot = Uniform(low=par_ranges['stot'][0], high=par_ranges['stot'][1], optguess=1000)
    fwm = Uniform(low=par_ranges['fwm'][0], high=par_ranges['fwm'][1], optguess=0.5)
    flm = Uniform(low=par_ranges['flm'][0], high=par_ranges['flm'][1], optguess=0.5)
    c = Uniform(low=par_ranges['c'][0], high=par_ranges['c'][1], optguess=0.5)
    ex = Uniform(low=par_ranges['ex'][0], high=par_ranges['ex'][1], optguess=5)
    ki = Uniform(low=par_ranges['ki'][0], high=par_ranges['ki'][1], optguess=0.5)
    kg = Uniform(low=par_ranges['kg'][0], high=par_ranges['kg'][1], optguess=0.5)
    ci = Uniform(low=par_ranges['ci'][0], high=par_ranges['ci'][1], optguess=0.5)
    cg = Uniform(low=par_ranges['cg'][0], high=par_ranges['cg'][1], optguess=0.5)

    # initialize the model class m
    m = m_28_xinanjiang_12p_4s()

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