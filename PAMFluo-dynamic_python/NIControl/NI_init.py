import sys
sys.path.insert(0, "../")

from NIControl.DronpaClass import DronpaIntensity
from config_DAQ import *

def init_NI_handler(name, trigger_color, sample_rate, acq_time, _run):

    ni = DronpaIntensity()
    ni.experiment_folder(name)
    ni.p.save_folder = ni.save_folder
    ni.cop.save_folder = ni.save_folder
    ni.g.save_folder = ni.save_folder
    ni.p.mongo = True
    ni.cop.mongo = True
    ni.g.mongo = True
    ni.p.mongo_run = _run
    ni.cop.mongo_run = _run
    ni.g.mongo_run = _run

    ni.generator_analog_channels = []
    ni.generator_digital_channels = []
    ni.detector_analog_channels = [detector_analog[intensity], detector_analog[fluorescence]]
    ni.detector_digital_channels = [detector_digital[i] for i in [arduino_blue, arduino_green, arduino_purple]]
    ni.trig_chan_detector = trigger[trigger_color]
    ni.trig_chan_detector_edge = 'RISING'



    ni.sample_rate = sample_rate #Hz
    ni.acq_time = acq_time #s
    ni.reading_samples = int(ni.sample_rate * ni.acq_time)

    ni.detector_init()
    ni.trigger_init()
    return ni