
def interflow_1(p1, S, Smax, flux):
    # interflow based on incoming flux size
    out = p1 * S / Smax * flux
    return out
