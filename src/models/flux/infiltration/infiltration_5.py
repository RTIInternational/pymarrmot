def infiltration_5(p1, p2, S1, S1max, S2, S2max):
	
	"""
	Flux function
	Description:  Maximum infiltration rate non-linearly based on relative deficit and storage
	Constraints:  S2 >= 0     - prevents complex numbers
	              f <= 10^9   - prevents numerical issues with Inf outcomes
	@(Inputs):    p1    - base infiltration rate [mm/d]
	              p2    - exponential scaling parameter [-]
	              S1    - current storage in S1 [mm]
	              S1max - maximum storage in S1 [mm]
	              S2    - current storage in S2 [mm]
	              S2max - maximum storage in S2 [mm]
	"""
	import numpy as np

	# prevents issues with small negative S2 values, as well as mathematical errors when S2 = 0
	# If S2/S2max = 1.0, scaler =  1
	# If S2/S2max = 0.5, scaler >  1
	# If S2/S2max = 0.1, scaler >> 1
	# Thus, 
	# if S2/S2max = 0.00, scaler >>> 1
	# However, Python does not calculate 0**-1 (DIV0 error) and we need the code below
	if S2 / S2max > 0:
		#We need to set the dtype to avoid “<error>”
		scaler = np.power(S2/S2max, -1*p2, dtype=np.float64)	
	else:
		scaler = 10**9 # Something large, but we cannot compute (0/S2max)**(-1*p2)
	
	# Resume function calculation
	# max(0,x) prevents issues with small negative values for S1
	out = max(0, p1 * (1 - S1 / S1max) * scaler)
	return out