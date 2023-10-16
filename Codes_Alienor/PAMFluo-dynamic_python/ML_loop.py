from ML_calibration import ex
import numpy as np
import time

N = 5

for limit_green in np.linspace(10, 200, N):
        r = ex.run(config_updates={'limit_green': int(limit_green)})
time.sleep(100)