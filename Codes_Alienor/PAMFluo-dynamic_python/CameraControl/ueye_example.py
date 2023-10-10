
""" 
uEye CCD main program

@fcn_header:
    def process_image(self, image_data, enable):
        --> custom function (can do save or some process)
    
    def main():

@refer: pyueye_example, source code from 
    https://en.ids-imaging.com/techtipps-detail/en_techtip-embedded-vision-kit.html

    Copyright (c) 2017 by IDS Imaging Development Systems GmbH.
    All rights reserved.

@create data: 2019.11.04
@update data: 2019.11.05
@author: Yang-Jie Gao
@e-mail: 60777001h@ntnu.edu.tw
"""

from ueye_camera import Camera
from ueye_utils import FrameThread
from ueye_gui import PyuEyeQtApp, PyuEyeQtView
from PyQt5 import QtGui

from pyueye import ueye

import glob
import cv2
import numpy as np

import os


if __name__ == "__main__":


    # camera class to simplify uEye API access
    cam = Camera()
    cam.init()
    cam.set_colormode(ueye.IS_CM_SENSOR_RAW8)
    cam.set_aoi(0,0, 1280, 1024)
    # cam.set_full_auto()
    cam.set_FrameRate(1)
    cam.set_Exposure(1000)
    cam.set_Gain(100)
    cam.alloc()
    cam.capture_video()

    # a thread that waits for new images and processes all connected views
    thread = FrameThread(cam, 15)
    thread.start()
    thread.join()
    cam.stop_video()
    cam.exit()

