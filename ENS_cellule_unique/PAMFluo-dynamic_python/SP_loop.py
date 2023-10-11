from SP_calibration import ex
import numpy as np
import time

N = 3

for length_SP in np.linspace(100, 300, N):
        r = ex.run(config_updates={'length_SP': int(length_SP)})
time.sleep(100)