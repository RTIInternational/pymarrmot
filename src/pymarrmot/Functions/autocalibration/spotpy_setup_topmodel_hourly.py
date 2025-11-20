from spotpy.parameter import Uniform
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import rmse

from pymarrmot.models.models.m_14_topmodel_7p_2s import m_14_topmodel_7p_2s

import pandas as pd
import numpy as np

class spotpy_setup(object):
    # Following example from https://spotpy.readthedocs.io/en/latest/How_to_link_a_model_to_SPOTPY/
    # Linking a model to SPOTPY is done by following five consecutive steps,
    # which are grouped in a spotpy_setup class

    # Step 1: Define the parameters of the model as class parameters
    par_ranges = {
        'suzmax': [1, 2000],    # suzmax, Maximum soil moisture storage in unsaturated zone [mm]
        'st': [0.05, 0.95],     # st, Threshold for flow generation and evap change as fraction of suzmax [-]  
        'kd': [0, 1],           # kd, Leakage to saturated zone flow coefficient [mm/d]
        'q0': [0.1, 200],       # q0, Zero deficit base flow speed [mm/d]
        'f': [0, 1],            # f, Baseflow scaling coefficient [mm-1]
        'chi': [1, 7.5],        # chi, Gamma distribution parameter [-]
        'phi': [0.1, 5]         # phi, Gamma distribution parameter [-]
        }
    
    # parameters that will be calibrated by Spotpy
    suzmax = Uniform(low=par_ranges['suzmax'][0], high=par_ranges['suzmax'][1], optguess=1000)  
    st = Uniform(low=par_ranges['st'][0], high=par_ranges['st'][1], optguess=0.5)               
    kd = Uniform(low=par_ranges['kd'][0], high=par_ranges['kd'][1], optguess=0.5)               
    q0 = Uniform(low=par_ranges['q0'][0], high=par_ranges['q0'][1], optguess=100)               
    f = Uniform(low=par_ranges['f'][0], high=par_ranges['f'][1], optguess=0.5)                  
    chi = Uniform(low=par_ranges['chi'][0], high=par_ranges['chi'][1], optguess=4.25)           
    phi = Uniform(low=par_ranges['phi'][0], high=par_ranges['phi'][1], optguess=2.55)           
    
    # class variables
    m = m_14_topmodel_7p_2s()
    m.delta_t = 1/24  # time step in days (1 hour)
    trueObs = None

    # Step 2: Write the def init function, which takes care of any things which need to be done only once
    def __init__(self,obj_func=None):
        self.obj_func = obj_func
        
        # Initial storage values
        input_s0 = np.array([1,  # SUZ [mm] is the current storage in the combined unsaturated zone and root zone
                     1])         # SSZ [mm] is the saturated zone storage


        #Define the solver settings
        input_solver_opts = {
            'resnorm_tolerance': 0.1,
            'resnorm_maxiter': 6
        }

        self.m.S0 = np.array(input_s0)
        self.m.solver_opts = input_solver_opts
        
        #USGS 03463300 - SOUTH TOE RIVER NEAR CELO, NC - usgsbasin-03463300_combined.parquet
        #USGS 02138500 - LINVILLE RIVER NEAR NEBO, NC - usgsbasin-02138500_combined.parquet
        #USGS 03441000 - DAVIDSON RIVER NEAR BREVARD, NC - usgsbasin-03441000_combined.parquet
        #USGS 03479000 - WATAUGA RIVER NEAR SUGAR GROVE, NC - usgsbasin-03479000_combined.parquet
        #USGS 03456100 - WEST FORK PIGEON RIVER AT BETHEL, NC - usgsbasin-03456100_combined.parquet

        df = pd.read_parquet('C:/Users/ssheeder/Repos/pymarrmot/forcing/pymarrmot/combined_forcing/12_year/usgsbasin-03463300_combined.parquet')

        # Create a climatology data input structure
        input_climatology = {
            'dates': df['value_time'].to_numpy(),  
            'precip': df['precip_mm'].to_numpy(),     
            'temp': df['temp_c'].to_numpy(),       
            'pet': df['pet_mm'].to_numpy()      
        }
        self.trueObs = df['discharge_mm'].tolist()
        self.m.input_climate = input_climatology
            
    # Step 3: Write the def simulation function, which starts your model and returns the results
    def simulation(self,x):
        #update theta with the new parameter set
        self.m.theta = np.array(x)
        
        #Here the model is actually started with a unique parameter combination that it gets from spotpy for each time the model is called
        (output_ex, output_in, output_ss, output_waterbalance) = self.m.get_output(nargout=4)
        #output_ex = self.m.get_output(nargout=1)
        return output_ex['Q'].tolist()
    
    # Step 4: Write the def evaluation function, which returns the observations
    def evaluation(self):
        return self.trueObs
    
    # Step 5: Write the def objectivefunction function, which returns the objective function value
    def objectivefunction(self, simulation, evaluation, params=None):
    #SPOTPY expects to get one or multiple values back, 
    #that define the performance of the model run
       #add simulation and evaluation lists to dataframe
        simulation_df = pd.DataFrame({'value_time': self.m.input_climate['dates'], 'simulated_discharge_mm': simulation})
        evaluation_df = pd.DataFrame({'value_time': self.m.input_climate['dates'], 'discharge_mm': evaluation})
       #left join on 'value_time' to align simulation and evaluation data
        merged_df = pd.merge(evaluation_df, simulation_df, on='value_time', how='left')
        evaluation_array = merged_df['discharge_mm'].to_numpy()
        simulation_array = merged_df['simulated_discharge_mm'].to_numpy()

        if not self.obj_func:
            # This is used if not overwritten by user
            score = rmse(evaluation_array, simulation_array)
            like = score
        else:
            # Spotpy minimizes the objective function, so for objective functions where fitness improves with increasing result, we need to multiply by -1
            
            # calculation of kge-lowflow (based on kge being selected as objective function)
            score = self.obj_func(evaluation_array, simulation_array)
            # eval_inverse = [1000 if (i<=0) else 1/i for i in evaluation]
            # sim_inverse = [1000 if (i<=0) else 1/i for i in simulation]
            # score2 = self.obj_func(eval_inverse, sim_inverse)
            # result = (score + score2)/2
            like = -1*score
        return like