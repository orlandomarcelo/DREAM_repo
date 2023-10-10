import numpy as np
import matplotlib.pyplot as plt

start = 5
stop = 25
offset = 1
sat = 3
writing_rate = 20 #Hz
acq_time = 30
writing_samples = int(acq_time*writing_rate)

def fitness_sequence(ni, offset, sat, start, stop, duration_sat):
    sample = np.linspace(0, ni.acq_time, num = ni.writing_samples , endpoint=False)
    actinic = np.zeros(sample.shape)
    actinic[start*ni.writing_rate:stop*ni.writing_rate] = offset
    pulse = np.zeros(sample.shape)
    pulse[stop*ni.writing_rate:stop*ni.writing_rate + int(duration_sat*ni.writing_rate)] = sat

    signal = pulse + actinic

    return signal


#############################
from VoltageIntensityClass import VoltageIntensity
from statsmodels.regression import linear_model
import numpy as np
from statsmodels.api import add_constant


V = VoltageIntensity(folder = "Experiments/2023-01-30_10_47_bode_diagram")

x = np.linspace(0.05, 4.5)

int = V.get_intensity_voltage("blue", 1, x)*1e6


a480 = 82
a532=7.7
x=x*100
int = int*a480/a532

def regression_affine(X, Y, details = True):
        Xreg = add_constant(X) #Add a constant to fit tan affine model

        model = linear_model.OLS(Y, Xreg) #Linear regression
        results = model.fit()
        [b, a] = results.params #parameters of the affine curve
        Yreg = a*X + b #regression curve

        return Yreg, a, b, results
    
   
Yreg, a, b, results = regression_affine(x[0:10], int[0:10], details = True) 


plt.plot(x, int, '.')

plt.plot(x[0:10], Yreg, '-')



list_green = np.array([0, 40, 90, 130, 230, 320, 570, 840, 1000, 1170])

volt = (list_green-b)/a

plt.plot(volt, a*volt + b, "ok")

plt.show()


##################################################


if True: 
    from mvgavg import mvgavg
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    import skimage
    path = "E:/Experimental_data/DREAM_microscope/"
    exp = "2023-10-10_11_57_Oscillation_Protocol/"
    list_output = glob.glob(path + exp + "*output*")
    list_time = glob.glob(path + exp + "*time*")
    for i in range(len(list_output)):
        
        fluo = np.load(list_output[i], allow_pickle=True)[0]
        time_array = np.load(list_time[i])
        x = mvgavg(time_array, 100)
        fluo = mvgavg(fluo, 100)

        plt.plot(x, fluo)
    plt.legend()

    plt.figure()
    list_output = glob.glob(path + exp + "*video.tiff*")
    list_time = glob.glob(path + exp + "*timing*")
    for i in range(len(list_output)):
        
        video = skimage.io.imread(list_output[i])
        time_array = np.load(list_time[i])
        fluo = np.mean(video, axis = (1,2))
        x = time_array-time_array[0]

        plt.plot(x, fluo, "-")
        plt.plot(x, fluo, ".k")
    plt.legend()
    plt.show()
    


import incense
from incense import ExperimentLoader
import pickle 
import os

def get_mongo_uri():
    in_devcontainer = (
        os.environ.get("TERM_PROGRAM") == "vscode"
        or os.environ.get("HOME") == "/home/vscode"
        or (os.environ.get("PATH") or "").startswith("/home/vscode")
    )
    if in_devcontainer:
        return "mongodb://mongo:27017"
    else:
        return None
    
loader = ExperimentLoader(
    mongo_uri=get_mongo_uri(), 
    db_name='sacred'
)

exp = loader.find_latest()

artifacts = exp.artifacts
keys = list(artifacts.keys())

output = artifacts[keys[0]].render()

video = pickle.load(exp.artifacts['video.npy'].file)
print(video.shape)
plt.imshow(video[10])
plt.show()
time.sleep(2)

##############################################################################


if True: 
    from mvgavg import mvgavg
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    import skimage
    path = "E:/Experimental_data/DREAM_microscope/"
    exp = "2023-10-10_11_57_Oscillation_Protocol/"
    list_output = glob.glob(path + exp + "*output*")
    list_time = glob.glob(path + exp + "*time*")
    for i in range(len(list_output)):
        
        fluo = np.load(list_output[i], allow_pickle=True)[0]
        time_array = np.load(list_time[i])
        x = mvgavg(time_array, 100)
        fluo = mvgavg(fluo, 100)

        plt.plot(x, fluo)
    plt.legend()

    plt.figure()
    list_output = glob.glob(path + exp + "*video.tiff*")
    list_time = glob.glob(path + exp + "*timing*")
    for i in range(len(list_output)):
        
        video = skimage.io.imread(list_output[i])
        time_array = np.load(list_time[i])
        fluo = np.mean(video, axis = (1,2))
        x = time_array-time_array[0]

        plt.plot(x, fluo, "-")
        plt.plot(x, fluo, ".k")
    plt.legend()
    plt.show()