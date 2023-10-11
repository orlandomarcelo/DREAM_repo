import pandas as pd
from alienlab.init_logger import logger
from tqdm import tqdm
import time as TIMING
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from mvgavg import mvgavg
import datetime
import tifffile as tiff

import cv2

from ArduinoControl import python_comm

from config_DAQ import *
from ArduinoControl.python_serial import *

from sacred.observers import MongoObserver
from sacred import Experiment
from Ingredients import pulse_ingredient, read_pulses
from Initialize_setup import start_and_stop, initialize, close_all_links, init_camera
import ipdb

from ThorlabsControl.KPZ101 import KPZ101

ex = Experiment('autofocus_wider', ingredients=[pulse_ingredient, start_and_stop])

ex.observers.append(MongoObserver())

ex.logger = logger
ex.add_config("config.json")

@ex.config
def my_config():

    name = "autofocus_wider"
    acq_time = 14*60
    actinic_filter = 1
    move_plate = False
    length_SP = 200
    period_SP= 10*sec
    limit_purple = 100
    limit_blue = 30
    trigger_color = "no_LED_trig"
    sample_rate=1e5
    exposure = 800
    gain  = 100


@ex.automain
def send_IBPC(_run, name, limit_blue, gain, exposure, limit_green, limit_purple, limit_red, trigger_color, sample_rate, acq_time, period_SP, length_SP, period_ML, actinic_filter, move_plate):


    def grab_image(exposure, num_frames, gain):
        thread, cam = init_camera(exposure = exposure, num_frames = num_frames, gain =  gain)

        # Acquisition
        ni.detector_start()
        
        thread.start()
        time.sleep(1)
        thread.stop_thread = True
        thread.join()
        cam.exit()
        try: #color camera vs grey
            L, H = cam.video[0].shape
            video = np.array(cam.video)

        except:
            L, H, d = cam.video[0].shape
            video = np.array(cam.video, dtype = "uint16")[:,:,:, 2]
        return video
    
    
    
    def show_images(images, cols = 1, titles = None):
        """Display a list of images in a single figure with matplotlib.
        
        Parameters
        ---------
        images: List of np.arrays compatible with plt.imshow.
        
        cols (Default = 1): Number of columns in figure (number of rows is 
                            set to np.ceil(n_images/float(cols))).
        
        titles: List of titles corresponding to each image. Must have
                the same length as titles.
        """
        assert((titles is None)or (len(images) == len(titles)))
        n_images = len(images)
        if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        #plt.show()
    
    # LED control tool DC4104 Thorlabs
    logger.name = name


    all_links = initialize(logger = logger, name = name, _run = _run, limit_blue = limit_blue, limit_green = limit_green, 
                                limit_purple = limit_purple, limit_red = limit_red, trigger_color = trigger_color, 
                                sample_rate = sample_rate, acq_time = acq_time, set_piezo = True)

    ctrlLED = all_links[0]
    ni = all_links[1]
    link_arduino = all_links[2]
    fwl = all_links[3]
    motors = all_links[4]

    fwl.move_to_filter(filters[actinic_filter])
    

    KP = all_links[5]
    KP.SetOutputVoltage(35)
    

    def focus(min, max, step, link_arduino, exposure, gain):
        python_comm.move_dz(motors, min)
        TIMING.sleep(2)
        add_master_digital_pulse(link_arduino, pins['no_LED_trig'], 0, 1000, 20, 0) #trigger
        add_digital_pulse(link_arduino, pins["blue"], 0, 20*minute, 20*minute, 0)

        positions = []
        position = min 
        start_measurement(link_arduino)
        images = []
        while position < max:
            print(position)
            python_comm.move_dz(motors, step)
            position += step
            positions.append(position)
            TIMING.sleep(5)
            images.append(grab_image(exposure, 1, gain)[0])
        
        stop_measurement(link_arduino)
    
        
        show_images(images, cols = 3)
        #ipdb.set_trace()
        blurs = []
        plt.figure()
        for i in range(len(images)):
            blur =  cv2.Laplacian(images[i][50:400, 400:800], cv2.CV_64F).var()
            plt.scatter(positions[i], blur)
            blurs.append(blur)
        
        plt.xlabel("Voltage piezo")
        plt.ylabel("Laplacian variance")
        #plt.show()

        
        y = np.argmax(blurs)
        print(y)
        print(positions)
        print(blurs)

        #position at max:
        python_comm.move_dz(motors, positions[y]-positions[-1])
        
        return min//2,  max//2, positions, blurs, images
        
    min = -1000
    max = 1000
    steps = 300
    plt.figure()
    voltage_list = []
    blur_list = []
    image_list = []
    for i in range(2):
        min, max, v, b, im = focus(min, max, steps, link_arduino, exposure, gain)
        voltage_list.extend(v)
        blur_list.extend(b)
        image_list.extend(im)
        steps=steps//2
    #ipdb.set_trace()
        
        
    with open("G:/DREAM/from_github/PAMFluo/specs/focus_wide.txt", 'w') as f:
        f.write('%f' %voltage_list[np.argmax(blur_list)])
    
    pos = voltage_list[-1]
    best_pos = voltage_list[np.argmax(blur_list)]
    
    close_all_links(*all_links)
    
    for i in range(len(voltage_list)):
            _run.log_scalar("blur", blur_list[i], voltage_list[i])

    
    ni.p.xlabel = "Voltage (V)"
    ni.p.ylabel = "Laplacian variance"
    ni.p.save_name = "autofocus" 
    fig = ni.p.plotting(np.array(voltage_list), np.array(blur_list))
    ni.p.saving(fig)
    _run.add_artifact(ni.p.save_path + ni.p.extension)


    ni.g.save_name = "image_focus"    
    fig = ni.g.multi(image_list[np.argmax(blur_list)])
    ni.g.saving(fig)
    _run.add_artifact(ni.g.save_path + ni.g.extension)

    
    
    
    
    #ref: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    #ref: https://github.com/antonio490/Autofocus
    #ref: https://sites.google.com/site/cuongvt101/research/Sharpness-measure
    
