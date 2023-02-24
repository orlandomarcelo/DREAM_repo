import numpy as np
from sacred import Ingredient
from config_DAQ import *
import matplotlib.pyplot as plt
import ipdb

from ThorlabsControl.DC4100_LED import ThorlabsDC4100
from ThorlabsControl.FW102 import FW102C
from ThorlabsControl.KPZ101 import KPZ101

from serial import *
from NIControl.NI_init import init_NI_handler
from ArduinoControl.python_serial import *
from ArduinoControl.python_comm import *
from pyueye import ueye


from CameraControl.ueye_camera import Camera
from CameraControl.ueye_utils import FrameThread


import threading
import queue

#ref: https://www.programmerall.com/article/57461489186/
class np_encode(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

"""basic bricks for experiments"""


start_and_stop = Ingredient('start_and_stop')


@start_and_stop.config
def cfg():
    port_DC4100 = "COM5"
    port_arduino = "COM4"
    port_filter_wheel = "COM3"
    port_motors = "COM6"
    initial_filter = filters[0]
    x=0
    y=0
    subsampling_factor=1
    height = 1936
    width = 1216
    gain = 100
    clock=118
    trigger = True
    binning_factor = 1
    trigger_frequency=1
    
@start_and_stop.capture
def initialize(name, logger, _run, limit_blue, limit_green, limit_purple, limit_red,
                trigger_color, sample_rate, acq_time, 
                port_arduino, port_DC4100, port_filter_wheel, initial_filter, port_motors, set_piezo = False):


    def init_DC4100(que):
        ctrlLED = ThorlabsDC4100(logger, port = port_DC4100)
        ctrlLED.initialise_fluo()
        ctrlLED.set_user_limit(LED_blue, limit_blue)
        ctrlLED.set_user_limit(LED_green, limit_green)
        ctrlLED.set_user_limit(LED_purple, limit_purple)
        ctrlLED.set_user_limit(LED_red, limit_red)

        que.put(ctrlLED)

    def init_arduino(que):
        link = Serial(port_arduino, 115200)
        time.sleep(2.0)
        reset_arduino(link)
        que.put(link)

    def init_ni(que):
        ni = init_NI_handler(name, trigger_color, sample_rate, acq_time, _run)
        with open(ni.save_folder + '/config.json', 'w') as fp:
            json.dump(_run.config, fp, cls=np_encode)
        que.put(ni)

    def init_wheel(que):
        fwl = FW102C(port=port_filter_wheel)
        fwl.move_to_filter(initial_filter)
        fwl.filter = initial_filter
        que.put(fwl)


    def init_motors(que):
        #motors = Serial(port_motors, 115200)
        #time.sleep(2.0)
        #reset_arduino(motors)
        motors = None
        que.put(motors)
        
    
    def init_piezo(que):
        KP = KPZ101(b"29250314")
        KP.BuildDeviceList()
        KP.Open()
        #KP.GetMaxVoltage()
        with open("G:/DREAM/from_github/PAMFluo/specs/focus.txt", 'r') as file:
            voltage = np.float(file.readline())
        KP.SetOutputVoltage(voltage)
        que.put(KP)



    if set_piezo == False:
        queues = [queue.Queue()  for i in range(5)]
        
    else:
        queues = [queue.Queue()  for i in range(6)]

    # Filter wheel
    t4 = threading.Thread(target = init_wheel, args = [queues[3],])
    t4.start()


    # LED control tool DC4104 Thorlabs
    t1 = threading.Thread(target = init_DC4100, args = [queues[0],])
    t1.start()
 
     
    # Upload code to the Arduino
    t3 = threading.Thread(target = init_arduino, args = [queues[2],])
    t3.start()
    t3.join()


    # NATIONAL INSTRUMENTS card SCB-68A
    t2 = threading.Thread(target = init_ni, args = [queues[1],])
    t2.start()
    
    
    # motorized_stage
    t5 = threading.Thread(target = init_motors, args = [queues[4],])
    t5.start()
    t5.join()
    
    if set_piezo == True:
        # piezo focus
        t6 = threading.Thread(target = init_piezo, args = [queues[5],])
        t6.start()
        t6.join()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    
    #ctrlLED, ni, link_arduino, fwl, motors = [q.get() for q in queues]


    return  [q.get() for q in queues]

@start_and_stop.capture
def close_all_links(*args):
    try:
        args[2].close()
    except:
        pass
    try: 
        args[3].close()
    except:
        pass
    try:
        args[0].disconnect()
    except:
        pass
    try:
        args[1].end_tasks()
    except:
        pass
    try:
        args[4].close()
    except:
        pass    
    try:
        args[5].SetOutputVoltage(0)
        args[5].Close()
    except:
        pass    

@start_and_stop.capture
def init_camera(exposure, num_frames, height, width, gain, trigger, x, y, subsampling_factor, binning_factor,  clock , trigger_frequency):
    cam = Camera()
    cam.init()
    cam.set_pixel_clock(clock)
    cam.set_colormode(ueye.IS_CM_SENSOR_RAW8)
    cam.set_binning(v=binning_factor, h=binning_factor)
    cam.height = height
    cam.width = width
    
    #cam.set_subsampling(v=subsampling_factor, h=subsampling_factor)
    cam.set_aoi(x,y, height, width)
    cam.get_aoi()
    # cam.set_full_auto()
    cam.set_trigger_delay(15)
    cam.set_FrameRate(trigger_frequency)
    cam.set_Exposure(exposure)
    cam.set_Gain(gain)
    cam.alloc()
    
    # a thread that waits for new images and processes all connected views
    thread = FrameThread(cam, num_frames, trigger)
    return thread, cam