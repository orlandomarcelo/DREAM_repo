
""" 
Provide utils to ueye_camera.py

@fcn_header:
    def get_bits_per_pixel(color_mode):
    class uEyeException(Exception):
    def check(ret):
    class ImageBuffer:
    class MemoryInfo:
    class ImageData:
    class Rect:
    class FrameThread(Thread):

@refer: pyueye_example, source code from 
    https://en.ids-imaging.com/techtipps-detail/en_techtip-embedded-vision-kit.html

    Copyright (c) 2017 by IDS Imaging Development Systems GmbH.
    All rights reserved.

@create data: 2019.11.04
@update data: 2019.11.05
@author: Yang-Jie Gao
@e-mail: 60777001h@ntnu.edu.tw
"""

from pyueye import ueye
from threading import Thread
from ctypes import byref
import cv2
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def get_bits_per_pixel(color_mode):
    """
    returns the number of bits per pixel for the given color mode
    raises exception if color mode is not is not in dict
    """
    
    return {
        ueye.IS_CM_SENSOR_RAW8: 8,
        ueye.IS_CM_SENSOR_RAW10: 16,
        ueye.IS_CM_SENSOR_RAW12: 16,
        ueye.IS_CM_SENSOR_RAW16: 16,
        ueye.IS_CM_MONO8: 8,
        ueye.IS_CM_RGB8_PACKED: 24,
        ueye.IS_CM_BGR8_PACKED: 24,
        ueye.IS_CM_RGBA8_PACKED: 32,
        ueye.IS_CM_BGRA8_PACKED: 32,
        ueye.IS_CM_BGR10_PACKED: 32,
        ueye.IS_CM_RGB10_PACKED: 32,
        ueye.IS_CM_BGRA12_UNPACKED: 64,
        ueye.IS_CM_BGR12_UNPACKED: 48,
        ueye.IS_CM_BGRY8_PACKED: 32,
        ueye.IS_CM_BGR565_PACKED: 16,
        ueye.IS_CM_BGR5_PACKED: 16,
        ueye.IS_CM_UYVY_PACKED: 16,
        ueye.IS_CM_UYVY_MONO_PACKED: 16,
        ueye.IS_CM_UYVY_BAYER_PACKED: 16,
        ueye.IS_CM_CBYCRY_PACKED: 16,        
    } [color_mode]


class uEyeException(Exception):
    def __init__(self, error_code):
        self.error_code = error_code
    def __str__(self):
        return "Err: " + str(self.error_code)


def check(ret):
    if ret != ueye.IS_SUCCESS:
        raise uEyeException(ret)


def fig_to_im(fig):
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        couple =  fig.canvas.get_width_height()[::-1] 
        img  = img.reshape((couple[0], couple[1], 3))
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        return img

class ImageBuffer:
    def __init__(self):
        self.mem_ptr = ueye.c_mem_p()
        self.mem_id = ueye.int()


class MemoryInfo:
    def __init__(self, h_cam, img_buff):
        self.x = ueye.int()
        self.y = ueye.int()
        self.bits = ueye.int()
        self.pitch = ueye.int()
        self.img_buff = img_buff

        rect_aoi = ueye.IS_RECT()
        check(ueye.is_AOI(h_cam,
                          ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ueye.sizeof(rect_aoi)))
        self.width = rect_aoi.s32Width.value
        self.height = rect_aoi.s32Height.value
        
        check(ueye.is_InquireImageMem(h_cam,
                                      self.img_buff.mem_ptr,
                                      self.img_buff.mem_id, self.x, self.y, self.bits, self.pitch))


class ImageData:
    def __init__(self, h_cam, img_buff):
        self.h_cam = h_cam
        self.img_buff = img_buff
        self.mem_info = MemoryInfo(h_cam, img_buff)
        self.color_mode = ueye.is_SetColorMode(h_cam, ueye.IS_GET_COLOR_MODE)
        self.bits_per_pixel = get_bits_per_pixel(self.color_mode)
        self.array = ueye.get_data(self.img_buff.mem_ptr,
                                   self.mem_info.width,
                                   self.mem_info.height,
                                   self.mem_info.bits,
                                   self.mem_info.pitch,
                                   True)

    def as_1d_image(self):        
        channels = int((7 + self.bits_per_pixel) / 8)
        import numpy
        if channels > 1:
            return numpy.reshape(self.array, (self.mem_info.height, self.mem_info.width, channels))
        else:
            return numpy.reshape(self.array, (self.mem_info.height, self.mem_info.width))

    def unlock(self):
        check(ueye.is_UnlockSeqBuf(self.h_cam, self.img_buff.mem_id, self.img_buff.mem_ptr))



def time_to_sec():
    dt = datetime.now()
    return dt.day*3600*24 + dt.hour*3600 +  dt.minute *60 + dt.second + 1e-6*dt.microsecond


class Rect:
    def __init__(self, x=0, y=0, width=0, height=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class FrameThread(Thread):
    def __init__(self, cam, num_frames, trigger = False, views=None, copy=True):
        super(FrameThread, self).__init__()
        self.timeout = 40000
        self.cam = cam
        self.cam.video = []
        self.running = True
        self.views = views
        self.copy = copy
        self.stop_thread = False
        self.num_frames = num_frames
        self.trigger = trigger

    def runold(self):
        while self.running:
            img_buffer = ImageBuffer()
            ret = ueye.is_WaitForNextImage(self.cam.handle(),
                                           self.timeout,
                                           img_buffer.mem_ptr,
                                           img_buffer.mem_id)
            if ret == ueye.IS_SUCCESS:
                self.notify(ImageData(self.cam.handle(), img_buffer))
            
            #break

    def run(self):
        self.cam.capture_video(self.trigger, wait=True)
        video = []
        timing = []
        start_run = time.time()
        i = 0
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        print(px)
        print(self.cam.width, self.cam.height)
        fig = plt.figure(figsize=((self.cam.height*px/2, self.cam.width*px/2)))
        #fig.canvas.draw()
        plt.xlabel("time(s)")
        plt.ylabel("fluorescence")

        while self.running:

            
            img_buffer = ImageBuffer()
            ret = ueye.is_WaitForNextImage(self.cam.handle(),
                                           self.timeout,
                                           img_buffer.mem_ptr,
                                           img_buffer.mem_id)


            #print(i)
            if ret == ueye.IS_SUCCESS:
                
                imdata = ImageData(self.cam.handle(), img_buffer)
                imdata.unlock()
        
                # bytes_per_pixel = int(nBitsPerPixel / 8)

                # ...reshape it in an numpy array...
                frame = imdata.as_1d_image()

                # ...resize the image by a half
                frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5)
                
            #---------------------------------------------------------------------------------------------------------------------------------------
                #Include image data processing here

            #---------------------------------------------------------------------------------------------------------------------------------------
                #print(frame.shape)

                self.im = frame[:,:]
                #im2 = skimage.filters.rank.autolevel(im ,selem = skimage.morphology.disk(3))
                #im2 = skimage.filters.rank.enhance_contrast(im ,selem = skimage.morphology.disk(3))
                #...and finally display it
                #cv2.imshow("SimpleLive_Python_uEye_OpenCV", im2)
                video.append(self.im)
                #plt.plot(np.mean(video, axis = (1,2)))                #im = fig_to_im(fig)
                #frame_tot = np.concatenate((frame, im[:,:,0]), axis=1) 

                cv2.imshow("SimpleLive_Python_uEye_OpenCV", frame)
                cv2.waitKey(1)
                # Press q if you want to end the loop

                #timing.append(time.time())#- start_run)
                timing.append(time_to_sec())
                i = i+1
                plt.clf()

            if self.stop_thread or i >= self.num_frames:
                print("im stopping thread right now. Num frames:", i)
                break

        cv2.destroyAllWindows()    
        self.cam.video = video
        self.cam.timing = timing


    def run_deep(self):
        
        self.cam.capture_video(self.trigger, wait=True)
        self.cam.set_colormode(ueye.IS_CM_SENSOR_RAW12)
        video = []
        timing = []
        start_run = time.time()
        i = 0
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        print(px)
        print(self.cam.width, self.cam.height)
        fig = plt.figure(figsize=((self.cam.height*px/2, self.cam.width*px/2)))
        #fig.canvas.draw()
        plt.xlabel("time(s)")
        plt.ylabel("fluorescence")

        while self.running:

            
            img_buffer = ImageBuffer()
            ret = ueye.is_WaitForNextImage(self.cam.handle(),
                                           self.timeout,
                                           img_buffer.mem_ptr,
                                           img_buffer.mem_id)


            #print(i)
            if ret == ueye.IS_SUCCESS:
                
                imdata = ImageData(self.cam.handle(), img_buffer)
                imdata.unlock()
        
                # bytes_per_pixel = int(nBitsPerPixel / 8)

                # ...reshape it in an numpy array...
                frame = imdata.as_1d_image()

                # ...resize the image by a half
                frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5)
                
            #---------------------------------------------------------------------------------------------------------------------------------------
                #Include image data processing here

            #---------------------------------------------------------------------------------------------------------------------------------------
                #print(frame.shape)


                frameSingle = ((frame[:,:,1])*256)+((frame[:,:,0]))
                self.im = frameSingle
                frameSingle8Bit = (frameSingle/16).astype(np.uint8)
                gray_BGR = cv2.cvtColor(frameSingle8Bit, cv2.COLOR_GRAY2BGR)
                gray_BGR = cv2.rectangle(gray_BGR, self.pt1, self.pt2, color=(0,255,0), thickness=5)
                gray_BGR = cv2.resize(gray_BGR,(0,0),fx=0.5, fy=0.5)
                #...and finally display it
                #cv2.imshow("SimpleLive_Python_uEye_OpenCV", im2)
                cv2.imshow("Camera Live Video", gray_BGR)
                cv2.waitKey(5)
                # Press q if you want to end the loop
                video.append(self.im)
                #timing.append(time.time())#- start_run)
                timing.append(time_to_sec())
                i = i+1

                plt.clf()

            if self.stop_thread or i >= self.num_frames:
                print("im stopping thread right now. Num frames:", i)
                break

        cv2.destroyAllWindows()    
        self.cam.video = video
        self.cam.timing = timing
    
    

    def notify(self, image_data):
        if self.views:
            if type(self.views) is not list:
                self.views = [self.views]
            for view in self.views:
                view.handle(image_data)

    def stop(self):
        self.cam.stop_video()
        self.running = False


# class RecThread(Thread):
    # def __init__(self, cam, views=None, copy=True):
        # super(FrameThread, self).__init__()
        # self.timeout = 1000
        # self.cam = cam
        # self.running = True
        # self.views = views
        # self.copy = copy

    # def run(self):
        # while self.running:
            # img_buffer = ImageBuffer()
            # ret = ueye.is_WaitForNextImage(self.cam.handle(),
                                           # self.timeout,
                                           # img_buffer.mem_ptr,
                                           # img_buffer.mem_id)
            # if ret == ueye.IS_SUCCESS:
                # self.notify(ImageData(self.cam.handle(), img_buffer))
            
            # break

    # def notify(self, image_data):
        # if self.views:
            # if type(self.views) is not list:
                # self.views = [self.views]
            # for view in self.views:
                # view.handle(image_data)

