import numpy as np
from seisloc.utils import find_closest

def mo_rates(times,duration,method="sin"):
    """
    Return a moment rate function given the duration and method.
    Moment rate function start from time 0 and normalzied to 1.
    """
    dt = times[1]-times[0]
    moRates = np.zeros_like(times)
    idx0,_ = find_closest(times,0)
    idx1,_ = find_closest(times,duration)
    if method == "sin":
        moRates[idx0:idx1] = np.sin(times[idx0:idx1]*np.pi/duration)

    return moRates
