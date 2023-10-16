# -*- coding: utf-8 -*-
"""
Simple example showing use of KPz101
Written 13/04/2021 by Jack Cater
"""

import os
import time
from ctypes import *
import sys
from enum import Enum

V = 75



# Emuns behave differently in Python, to access them you have to create a class like below
# To Enum value can be found in the respective header file for the controller
class PZ_ControlModeTypes(Enum):
    PZ_ControlModeUndefined = 0 # Undefined
    PZ_OpenLoop = 1 # Open Loop
    PZ_CloseLoop = 2 # Closed loop
    PZ_OpenLoopSmooth = 3 # Open loop with smoothing
    PZ_CloseLoopSmooth = 4 # Closed loop with smoothing

class KPZ101():
    def __init__(self, serial_number):
        # If you're using Python 3.7 or older comment add_dll_directory and uncomment chdir
        #os.chdir(r"C:/Program Files/Thorlabs/Kinesis")
        #os.add_dll_directory(r"C:\Program Files\Thorlabs\Kinesis")
        self.lib = cdll.LoadLibrary(r"C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.KCube.Piezo.dll")
        self.serial_number =  c_char_p(serial_number)


    def BuildDeviceList(self):
                # Build device list
            retval = self.lib.TLI_BuildDeviceList()
            if retval == 0:
                print("Device List Built successfully...")
            else:
                print("Error Code: {}", format(retval))
                sys.exit() # stop any further code from running

            # Print number of devices connected
            print("Number of devices = {}".format(self.lib.TLI_GetDeviceListSize()))


    def VoltageToMoveValue(self, volts):
        """To set the output voltage the second parameter is The voltage as a percentage of MaxOutputVoltage,
        range -32767 to 32767 equivalent to -100% to 100%. This converts this to a real voltage value"""
        _max_voltage = self.lib.PCC_GetMaxOutputVoltage(self.serial_number)
        return c_short(int((volts / _max_voltage) * 32767))

    def MoveValueToVoltage(self, bit_value):
        _max_voltage = self.lib.PCC_GetMaxOutputVoltage(self.serial_number)
        return (bit_value / 32767) * _max_voltage

    def GetMaxVoltage(self):
        max_voltage = self.lib.PCC_GetMaxOutputVoltage(self.serial_number)
        print("Max voltage: {}" .format(max_voltage/10))
        return max_voltage

    def Open(self):
        # Open the device
        if self.lib.PCC_Open(self.serial_number) == 0:
            # Start polling at 200ms intervals
            self.lib.PCC_StartPolling(self.serial_number, 200)
            # Enable voltage instructions
            self.lib.PCC_Enable(self.serial_number)

            time.sleep(2)

            # Set open loop mode
            self.lib.PCC_SetPositionControlMode(self.serial_number, PZ_ControlModeTypes.PZ_OpenLoop.value)

    def SetOutputVoltage(self, V):
            # Set output voltage
            self.lib.PCC_SetOutputVoltage(self.serial_number, self.VoltageToMoveValue(V*10))

            # sleep for an arbitrary time, in reality you would check to see if the command has finished
            time.sleep(5)

            # Print output voltage
            print("Voltage = {}".format(self.MoveValueToVoltage(self.lib.PCC_GetOutputVoltage(self.serial_number))/10))

    def Close(self):
            self.lib.PCC_Disable(self.serial_number)
            # Stop polling
            self.lib.PCC_StopPolling(self.serial_number)
            # Close device
            self.lib.PCC_Close(self.serial_number)


if __name__ == "__main__":
    KP = KPZ101(b"29250314")
    KP.BuildDeviceList()
    KP.Open()
    KP.GetMaxVoltage()
    KP.SetOutputVoltage(75)
    #KP.Close()
