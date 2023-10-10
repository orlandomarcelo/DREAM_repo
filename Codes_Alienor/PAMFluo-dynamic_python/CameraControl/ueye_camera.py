
""" 
uEye CCD base setup
e.g. colormode, capture image, open ccd, stop ccd

@fcn_header:
    class Camera:

@refer: pyueye_example, source code from 
    https://en.ids-imaging.com/techtipps-detail/en_techtip-embedded-vision-kit.html

    Copyright (c) 2017 by IDS Imaging Development Systems GmbH.
    All rights reserved.

@create data: 2019.11.04
@update data: 2019.11.05
@author: Yang-Jie Gao
@e-mail: 60777001h@ntnu.edu.tw
"""

import numpy as np
import tifffile as tiff


from pyueye import ueye
from CameraControl.ueye_utils import (uEyeException, Rect, get_bits_per_pixel,
                                  ImageBuffer, check)
import time

class Camera:
    def __init__(self, device_id=0):
        self.h_cam = ueye.HIDS(device_id)
        self.img_buffers = []

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, _type, value, traceback):
        self.exit()

    def handle(self):
        return self.h_cam

    def alloc(self, buffer_count=3):
        rect = self.get_aoi()
        bpp = get_bits_per_pixel(self.get_colormode())

        for buff in self.img_buffers:
            check(ueye.is_FreeImageMem(self.h_cam, buff.mem_ptr, buff.mem_id))

        for i in range(buffer_count):
            buff = ImageBuffer()
            ueye.is_AllocImageMem(self.h_cam,
                                  rect.width, rect.height, bpp,
                                  buff.mem_ptr, buff.mem_id)
            
            check(ueye.is_AddToSequence(self.h_cam, buff.mem_ptr, buff.mem_id))

            self.img_buffers.append(buff)

        ueye.is_InitImageQueue(self.h_cam, 0)

    def init(self):
        ret = ueye.is_InitCamera(self.h_cam, None)
        if ret != ueye.IS_SUCCESS:
            self.h_cam = None
            raise uEyeException(ret)
            
        return ret

    def exit(self):
        ret = None
        if self.h_cam is not None:
            ret = ueye.is_ExitCamera(self.h_cam)
        if ret == ueye.IS_SUCCESS:
            self.h_cam = None
            
    def return_video(self, save_folder, extend_name = ""):
        try: #color camera vs grey
            L, H = self.video[0].shape
            video = np.array(self.video)

        except:
            L, H, d = self.video[0].shape
            video = np.array(self.video, dtype = "uint16")[:,:,:, 2]
    
        tiff.imwrite(save_folder + "/" + extend_name + "video.tiff", video, photometric='minisblack')
        timing = np.array(self.timing)
        np.save(save_folder + "/" + extend_name + "video_timing.npy", timing)
        timing# -= timing[0]
        return video, timing, L, H 

    def get_aoi(self):
        rect_aoi = ueye.IS_RECT()
        ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
        print("x:%d, y:%d, height:%d, width:%d"%(rect_aoi.s32X.value,
                    rect_aoi.s32Y.value,
                    rect_aoi.s32Width.value,
                    rect_aoi.s32Height.value))
        
        return Rect(rect_aoi.s32X.value,
                    rect_aoi.s32Y.value,
                    rect_aoi.s32Width.value,
                    rect_aoi.s32Height.value)

    def set_aoi(self, x, y, width, height):
        rect_aoi = ueye.IS_RECT()
        rect_aoi.s32X = ueye.int(x)
        rect_aoi.s32Y = ueye.int(y)
        rect_aoi.s32Width = ueye.int(width)
        rect_aoi.s32Height = ueye.int(height)
        ret = ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
        if ret != ueye.IS_SUCCESS:
            raise uEyeException(ret)
        return ret

    def set_pixel_clock(self, clock):
        #instructioin
        #1 get min
        #4-5 get max
        #6 set (30, 59, 118, 237, 474)
        pc = ueye.INT(clock)
        nRet = ueye.is_PixelClock(self.h_cam, 6, pc, ueye.sizeof(pc))
        print(pc)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetPixelClock ERROR")        


    def capture_video(self, trigger=False, wait=True):
                # Activates the camera's live video mode (free run mode)
        if trigger==True:
            nRet = ueye.is_SetExternalTrigger(self.h_cam, ueye.IS_SET_TRIGGER_LO_HI) 
            if nRet != ueye.IS_SUCCESS:
                print("is_SetExternalTrigger ERROR")

        else:
            nRet = ueye.is_SetExternalTrigger(self.h_cam, ueye.IS_SET_TRIGGER_OFF)# ueye.IS_SET_TRIGGER_LO_HI) #
            if nRet != ueye.IS_SUCCESS:
                print("is_SetExternalTrigger ERROR")
                
        wait_param = ueye.IS_WAIT if wait else ueye.IS_DONT_WAIT
        return ueye.is_CaptureVideo(self.h_cam, wait_param)

    def stop_video(self):
        return ueye.is_StopLiveVideo(self.h_cam, ueye.IS_FORCE_VIDEO_STOP)
    
    def freeze_video(self, wait=False):
        wait_param = ueye.IS_WAIT if wait else ueye.IS_DONT_WAIT
        return ueye.is_FreezeVideo(self.h_cam, wait_param)

    def set_colormode(self, colormode):
        check(ueye.is_SetColorMode(self.h_cam, colormode))
        
    def get_colormode(self):
        ret = ueye.is_SetColorMode(self.h_cam, ueye.IS_GET_COLOR_MODE)
        return ret
        

    def set_trigger_delay(self, delay):

        ret = ueye.is_SetTriggerDelay(self.h_cam, ueye.INT(delay))
        if ret != ueye.IS_SUCCESS:
                print("is_SetExternalTrigger ERROR")

        print("DELAY µs:", ueye.is_SetTriggerDelay(self.h_cam, ueye.IS_GET_TRIGGER_DELAY))


    def set_FrameRate(self, rate):
        rate = ueye.DOUBLE(rate)
        newrate = ueye.DOUBLE()
        check(ueye.is_SetFrameRate(self.h_cam, rate, newrate))
        print('FR:',newrate)

    def set_Gain(self, gain):
        gain = ueye.INT(gain)
        check(ueye.is_SetHardwareGain(self.h_cam, gain, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER,
                                ueye.IS_IGNORE_PARAMETER))
        print('FR:',gain)
    
    def set_Exposure(self, val):
        ms = ueye.DOUBLE(val)
        check(ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, ms, ueye.sizeof(ms)))
        print('EXP:',ms)
        
    def set_full_auto(self):
        enable = ueye.DOUBLE(1)
        zero = ueye.DOUBLE(0)
        ms = ueye.DOUBLE(20)
        rate = ueye.DOUBLE(50)
        newrate = ueye.DOUBLE()

        ret = ueye.is_SetAutoParameter(self.h_cam, ueye.IS_SET_ENABLE_AUTO_GAIN, enable, zero)
        ret = ueye.is_SetAutoParameter(self.h_cam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, enable, zero)
        ret = ueye.is_SetFrameRate(self.h_cam, rate, newrate)
        ret = ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, ms, ueye.sizeof(ms))
        
        print("Auto mode")
        print('A_GAIN:',ret)
        print('A_SHUTTER:',ret)
        print('FR:',ret,newrate)
        print('EXP:',ret,ms)

    def get_format_list(self):
        count = ueye.UINT()
        check(ueye.is_ImageFormat(self.h_cam, ueye.IMGFRMT_CMD_GET_NUM_ENTRIES, count, ueye.sizeof(count)))
        format_list = ueye.IMAGE_FORMAT_LIST(ueye.IMAGE_FORMAT_INFO * count.value)
        format_list.nSizeOfListEntry = ueye.sizeof(ueye.IMAGE_FORMAT_INFO)
        format_list.nNumListElements = count.value
        check(ueye.is_ImageFormat(self.h_cam, ueye.IMGFRMT_CMD_GET_LIST,
                                  format_list, ueye.sizeof(format_list)))
        return format_list



##https://github.com/Kenrick-Trip/adaptiveoptics/blob/9cc38336cc51ce7086becb177e59ad4612be3012/cameras/ueye_camera.py

    def set_subsampling(self, v=1,h=1):
        val = ueye.IS_SUBSAMPLING_DISABLE
        if v==2:
            val |= ueye.IS_SUBSAMPLING_2X_VERTICAL
        elif v==4:
            val |= ueye.IS_SUBSAMPLING_4X_VERTICAL 

        elif v==8:
            val |= ueye.IS_SUBSAMPLING_8X_VERTICAL
        elif v==16:
            val |= ueye.IS_SUBSAMPLING_16X_VERTICAL 
            
        if h==2:
            val |= ueye.IS_SUBSAMPLING_2X_HORIZONTAL
        elif h==4:
            val |= ueye.IS_SUBSAMPLING_4X_HORIZONTA
            
        elif h==8:
            val |= ueye.IS_SUBSAMPLING_8X_HORIZONTAL
        elif h==16:
            val |= ueye.IS_SUBSAMPLING_16X_HORIZONTAL


        ret = ueye.is_SetSubSampling(self.h_cam, val)
        if ret != ueye.IS_SUCCESS:
            raise uEyeException(ret)
        else: print("Subsampling factor:", v)
        return ret

    
    def set_binning(self, v=1, h=1):

        val = ueye.IS_BINNING_DISABLE
        
        if v==2:
            val |= ueye.IS_BINNING_2X_VERTICAL
            
        if h==2:
            val |= ueye.IS_BINNING_2X_HORIZONTAL


        ret = ueye.is_SetBinning(self.h_cam, val)
        if ret != ueye.IS_SUCCESS:
            raise uEyeException(ret)
        else:
            print("Binning factor:", v)
        return ret