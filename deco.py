import time
import functools

def timer(func):
    functools.wraps(func)
    def wrapper_timer(*args,**kwargs):
        startTime = time.perf_counter()
        func(*args,**kwargs)
        endTime = time.perf_counter()
        runTime = endTime - startTime
        print(f"Finished {func.__name__!r} in {runTime:.4f} secs")
    return wrapper_timer
