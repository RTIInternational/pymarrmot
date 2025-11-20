"""
This file is part of the Python Modular Assessment of Rainfall-Runoff Models
Toolbox (pyMARRMoT).
pyMARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
WARRANTY. See <https://www.gnu.org/licenses/> for details.

Contact: ssheeder@rti.org

This example workflow contains an example application of autocalibration of a
single model to a single catchment. The model utilized in this example is the
Hymod model, which is a conceptual rainfall-runoff model with five stores and
five fluxes. The model is calibrated to a single catchment using the SCE-UA
optimization algorithm implemented in the SPOTPY package. 

This example file calls the 'spotpy_setup' class, which is a wrapper around
the Hymod model and the SPOTPY package. The 'spotpy_setup' class contains
methods to set up the model, define the parameter ranges, and define the
objective function. The 'spotpy_setup' class is used to set up the model for
calibration and to run the SCE-UA optimization algorithm.
"""
import os
import sys
import spotpy
import spotpy.objectivefunctions as of
from pymarrmot.functions.autocalibration.spotpy_setup_topmodel_hourly import spotpy_setup as setup
import matplotlib.pyplot as plt

if not os.path.exists('output'):
    os.makedirs('output')

#Set the number of model runs
reps = 10
spotpy_setup = setup(of.kge)
sampler=spotpy.algorithms.sceua(spotpy_setup, dbname='./output/' + 'SCEUA_topmodel_hourly_south_toe_river', dbformat='csv')
sampler.sample(reps) #, ngs=7, kstop=10, peps=0.001, pcento=0.001
results = spotpy.analyser.load_csv_results('./output/SCEUA_topmodel_hourly_south_toe_river')

#find the run_id with the minimal objective function value
bestindex,bestobjf = spotpy.analyser.get_minlikeindex(results)

#Then we select the best model run:
best_model_run = results[bestindex]

#And filter results for simulation results only:
if best_model_run is not None and hasattr(best_model_run, 'dtype'):
    fields=[word for word in best_model_run.dtype.names if word.startswith('sim')]
    best_simulation = list(best_model_run[fields])
else:
    print("best_model_run is None or has no dtype attribute")
    sys.exit()
    
df = spotpy_setup.evaluation()
if df is not None:
    eval_list = df['discharge_mm']
    eval_list = eval_list.tolist()
else:
    print("Evaluation data is None")
    sys.exit()

#Timeseries plot of the best model run
fig= plt.figure(figsize=(16,9))
ax = plt.subplot(1,1,1)
ax.plot(best_simulation,color='black',linestyle='solid', label='Best objf.='+str(bestobjf))
ax.plot(eval_list,color='red',markersize=3, label='Observation data')
plt.xlabel('Number of Observation Points')
plt.ylabel ('Discharge [mm]')
plt.legend(loc='upper right')
fig.savefig('SCEUA_topmodel_hourly_south_toe_river.png',dpi=300)

#FDC plot of the best model run, exceedance probability on the x-axis and discharge on the y-axis
#Sort the simulated and observed data, and calculate the exceedance probability
sim_sorted = best_simulation
sim_sorted.sort(reverse=True)
obs_sorted = eval_list
obs_sorted.sort(reverse=True)
exceedance_prob = [(x/(len(sim_sorted) + 1)*100) for x in range(len(sim_sorted))]
#get the FDC plot
fig= plt.figure(figsize=(16,9))
ax = plt.subplot(1,1,1)
ax.plot(exceedance_prob, sim_sorted, color='black', linestyle='solid', label='Simulated')
ax.plot(exceedance_prob, obs_sorted, color='red', linestyle='solid', linewidth= 2,label='Observed')
plt.yscale('log')
plt.xlabel('Exceedance Probability')
plt.ylabel('Discharge [mm]')
plt.legend(loc='upper right')
fig.savefig('SCEUA_topmodel_hourly_south_toe_river_FDC.png', dpi=300)