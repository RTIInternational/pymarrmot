from pymarrmot.functions.flux_smoothing.smooth_threshold_storage_logistic import smooth_threshold_storage_logistic

def evap_16(p1, S1, S2, S2min, Ep, dt):
    # Flux function
	# ------------------
	# Description:  Scaled evaporation if another store is below a threshold
	# Constraints:  f <= S1/dt
	# @(Inputs):    p1    - linear scaling parameter [-]
	#               S1    - current storage in S1 [mm]
	#               S2    - current storage in S2 [mm]
	#               S2min - threshold S2 storage for evaporation occurence [mm]
	#               Ep    - potential evapotranspiration rate [mm/d]
	#               dt    - time step size [d]
    
	out = min((p1 * Ep) * smooth_threshold_storage_logistic(S2, S2min), S1 / dt)
	return out
