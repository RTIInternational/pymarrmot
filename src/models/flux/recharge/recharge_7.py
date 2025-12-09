
def recharge_7(p1, fin):
    '''
    Constant recharge limited by incoming flux
    
    :param p1: maximum recharge rate [mm/d]
    :param fin: incoming flux [mm/d]
    :return: minimum of p1 and fin
    '''
    out = min(p1, fin)
    return out
