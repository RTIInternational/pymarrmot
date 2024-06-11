"""
This is a test to run models 11 to 20 with the same input data and solver options.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymarrmot.models.models.m_21_flexb_9p_3s import m_21_flexb_9p_3s
from pymarrmot.models.models.m_22_vic_10p_3s import m_22_vic_10p_3s
from pymarrmot.models.models.m_23_lascam_24p_3s import m_23_lascam_24p_3s
from pymarrmot.models.models.m_24_mopex1_5p_4s import m_24_mopex1_5p_4s
from pymarrmot.models.models.m_25_tcm_6p_4s import m_25_tcm_6p_4s
from pymarrmot.models.models.m_26_flexi_10p_4s import m_26_flexi_10p_4s
from pymarrmot.models.models.m_27_tank_12p_4s import m_27_tank_12p_4s
from pymarrmot.models.models.m_28_xinanjiang_12p_4s import m_28_xinanjiang_12p_4s
from pymarrmot.models.models.m_29_hymod_5p_5s import m_29_hymod_5p_5s
from pymarrmot.models.models.m_30_mopex2_7p_5s import m_30_mopex2_7p_5s

# 1. Prepare data
df = pd.read_csv('c:/users/ssheeder/repos/pymarrmot/examples/Example_DataSet.csv')

# Create a climatology data input structure
input_climatology = {
    'dates': df['Date'].to_numpy(),      # Daily data: date in ??? format
    'precip': df['Precip'].to_numpy(),   # Daily data: P rate [mm/d]
    'temp': df['Temp'].to_numpy(),       # Daily data: mean T [degree C]
    'pet': df['PET'].to_numpy(),         # Daily data: Ep rate [mm/d]
}

# 2. Define the model settings
model_list =   [m_21_flexb_9p_3s,
                m_22_vic_10p_3s,
                m_23_lascam_24p_3s,
                m_24_mopex1_5p_4s,
                m_25_tcm_6p_4s,
                m_26_flexi_10p_4s,
                m_27_tank_12p_4s,
                m_28_xinanjiang_12p_4s,
                m_29_hymod_5p_5s,
                m_30_mopex2_7p_5s]

# 3. Define the solver settings
input_solver_opts = {
    'resnorm_tolerance': 0.1,
    'resnorm_maxiter': 6
}

# Create and run all model objects
results_sampling = []

results_sampling.append(['model name', 'parameter_values', 'output_ex', 'output_in', 'output_ss', 'output_wb'])

for model in model_list:
    print(f"Now starting model {model}.")
    m = model()
    m.delta_t = 1
    model_range = np.array(m.par_ranges)
    num_par = m.num_params
    num_store = m.num_stores

    input_theta = model_range[:, 0] + np.random.rand(num_par) * (model_range[:, 1] - model_range[:, 0])
    input_s0 = np.zeros(num_store)

    output_ex, output_in, output_ss, output_waterbalance = m.get_output(4,input_climatology, input_s0, input_theta, input_solver_opts)

    results_sampling.append([model, input_theta, output_ex, output_in, output_ss, output_waterbalance])

# 6. Analyze the outputs
t = input_climatology['dates']
streamflow_observed = df['Q'].to_numpy()

plt.figure(figsize=(10, 6))
plt.plot(t, streamflow_observed, 'k', linewidth=2)
for i in range(len(model_list)):
    plt.plot(t, results_sampling[i + 1][2]['Q'])

plt.legend(['Observed'] + model_list, loc='upper right')
plt.title('Model sampling results')
plt.ylabel('Streamflow [mm/d]')
plt.xlabel('Time [d]')
plt.grid(True)
plt.xticks(rotation=45)
plt.ylim([0, 70])
plt.tight_layout()
plt.show(block=True)

