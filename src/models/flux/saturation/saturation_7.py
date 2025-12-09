
# Python translation of the MATLAB function saturation_7
# Function definition
def saturation_7(p1, p2, p3, p4, p5, S, In):
    from scipy.integrate import quad
    import numpy as np
    
    # Define the integrand function
    integrand = lambda x: 1 / (p1 * np.math.gamma(p2)) * (np.maximum(x - p3, 0) / p1) ** (p2 - 1) * np.exp(-1 * np.maximum(x - p3, 0) / p1)
    
    # Perform the integration
    out, _ = quad(integrand, p5 * max(S, 0) + p4, np.inf)
    
    return out * In
