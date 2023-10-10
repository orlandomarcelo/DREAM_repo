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
# If you're using Python 3.7 or older comment add_dll_directory and uncomment chdir
os.chdir(r"C:/Program Files/Thorlabs/Kinesis")
#os.add_dll_directory(r"C:\Program Files\Thorlabs\Kinesis")
lib = cdll.LoadLibrary(r"C:/Program Files/Thorlabs/Kinesis/Thorlabs.MotionControl.KCube.Piezo.dll")


def GetDeviceList():
    """A function to print the serial numbers of the connected devices"""
    serial_number_list = []
    listSize = lib.TLI_GetDeviceListSize()
    if listSize == 0:
        print("No Devices Connected...")
    else:
        # Devices found, next look for S/Ns by device type
        TL_cBufSize = 255
        devID = 29
        sBuf = c_buffer(TL_cBufSize)

        if lib.TLI_GetDeviceListByTypeExt(sBuf, TL_cBufSize, devID) != 0:
            print("No devices of type {} found".format(devID))
            return
        serial_number_list = sBuf.value.decode().rsplit(",")[0:-1]
    return serial_number_list

def VoltageToMoveValue(volts):
    """To set the output voltage the second parameter is The voltage as a percentage of MaxOutputVoltage,
    range -32767 to 32767 equivalent to -100% to 100%. This converts this to a real voltage value"""
    _max_voltage = lib.PCC_GetMaxOutputVoltage(serial_number)
    return c_short(int((volts / _max_voltage) * 32767))

def MoveValueToVoltage(bit_value):
    _max_voltage = lib.PCC_GetMaxOutputVoltage(serial_number)
    return (bit_value / 32767) * _max_voltage

# Emuns behave differently in Python, to access them you have to create a class like below
# To Enum value can be found in the respective header file for the controller
class PZ_ControlModeTypes(Enum):
    PZ_ControlModeUndefined = 0 # Undefined
    PZ_OpenLoop = 1 # Open Loop
    PZ_CloseLoop = 2 # Closed loop
    PZ_OpenLoopSmooth = 3 # Open loop with smoothing
    PZ_CloseLoopSmooth = 4 # Closed loop with smoothing

# Initialise simulations - Comment out if using actual hardware
#lib.TLI_InitializeSimulations()

# Build device list
retval = lib.TLI_BuildDeviceList()
if retval == 0:
    print("Device List Built successfully...")
else:
    print("Error Code: {}", format(retval))
    sys.exit() # stop any further code from running

# Print number of devices connected
print("Number of devices = {}".format(lib.TLI_GetDeviceListSize()))

# Print connected devices
#device_list = GetDeviceList()
#print(device_list)

# set up serial number variable
serial_number = c_char_p(b"29250314")
# Serial number value used for printing
sn_ptr = "yoyo" #device_list[0]

# Open the device
if lib.PCC_Open(serial_number) == 0:
    # Start polling at 200ms intervals
    lib.PCC_StartPolling(serial_number, 200)
    # Enable voltage instructions
    lib.PCC_Enable(serial_number)

    time.sleep(2)

    # Set open loop mode
    lib.PCC_SetPositionControlMode(serial_number, PZ_ControlModeTypes.PZ_OpenLoop.value)
    # Get max voltage
    max_voltage = lib.PCC_GetMaxOutputVoltage(serial_number)
    print("Max voltage: {}" .format(max_voltage/10))
    # Set output voltage
    lib.PCC_SetOutputVoltage(serial_number, VoltageToMoveValue(V*10))

    # sleep for an arbitrary time, in reality you would check to see if the command has finished
    time.sleep(5)

    # Print output voltage
    print("Device {} Voltage = {}".format(sn_ptr, MoveValueToVoltage(lib.PCC_GetOutputVoltage(serial_number))/10))

    lib.PCC_Disable(serial_number)
    # Stop polling
    lib.PCC_StopPolling(serial_number)
    # Close device
    lib.PCC_Close(serial_number)

# Comment if not using simulator
#lib.TLI_UninitializeSimulations()
