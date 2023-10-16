import sys
sys.path.insert(0, "G:\DREAM/from_github/CIFRE_DREAM")
from AcquisitionClass import Acquisition
from config_DAQ import *

import time
import serial
import argparse
import json
import traceback


def send_command(link, s):
    print("Command: %s" % s)
    command = "#" + s + ":xxxx\r"
    print(command)
    link.write(command.encode('ascii'))
    return assert_reply(read_reply(link))
    
def read_reply(link):
    while True:
        s = link.readline().decode("ascii").rstrip()
        if s[0] == "#":
            if s[1] == "!":
                print("Log: %s" % s)
            else:
                print("Reply: %s" % s)
                break;
    return s

def assert_reply(line):
    s = str(line)
    start = s.find("[")
    end = 1 + s.find("]")
    array_str = s[start:end]
    return_values = json.loads(array_str)

    print(return_values)
    status_code = return_values[0]
    success = (status_code == 0)
    if not success:
        raise RuntimeError(return_values[1]) 
    return return_values


    
def test_add_digital_pulse(link, pin, offset, period, duration):
    send_command(link, "d[%d,%d,%d,%d]" % (pin, offset, period, duration))

def test_start_measurement(link):
    send_command(link, "b")

def test_stop_measurement(link):
    send_command(link, "e")

def test_reset(link):
    link.setDTR(False) # Drop DTR
    time.sleep(0.022)    # Read somewhere that 22ms is what the UI does.
    link.setDTR(True)  # UP the DTR back
    link.close()
    link = serial.Serial("COM4", 115200)
    return link

    return link
def wait(link):
    while True:
        status = send_command(link, "I")
        if status[1] == 1:
            break
        time.sleep(1)
    
if __name__ == "__main__":

    link = serial.Serial("COM4", 115200)
    time.sleep(2.0)

        
    pam = Acquisition()
    pam.experiment_folder("test_arduino")
    pam.p.save_folder = pam.save_folder

    pam.generator_analog_channels = []
    pam.generator_digital_channels = []
    pam.detector_analog_channels = [detector_analog[intensity], detector_analog[fluorescence]]
    pam.detector_digital_channels =  [detector_digital[i] for i in [arduino_blue, arduino_green, arduino_purple]]
    pam.trig_chan_detector = trigger[arduino_green]
    pam.trig_chan_detector_edge = 'RISING'

    pam.sample_rate = 1e3 #Hz
    pam.acq_time = 60 #s
    pam.reading_samples = int(pam.sample_rate * pam.acq_time)

    pam.detector_init()
    pam.detector_start()




    print("Testing the Digital pulse")
    
    link = test_reset(link)
    time.sleep(2)

    test_add_digital_pulse(link, pins["green"], 0, 1000, 500)
    test_add_digital_pulse(link, pins["blue"], 10, 1000, 500)
    test_start_measurement(link)
    response = input("Is the blue light blinking ? [Y/n]")
    if (response == "n"):
        sys.exit("Check blue light cables")
    response = input("Is the green light blinking ? [Y/n]")
    if (response == "n"):
        sys.exit("Check green light cables")
    
    test_stop_measurement(link)
    link = test_reset(link)
    time.sleep(2)
    print("All tests OK")    

    test_add_digital_pulse(link, pins["green"], 0, 1000, 500)
    test_add_digital_pulse(link, pins["blue"], 10, 1000, 500)
    test_start_measurement(link)
    response = input("Is the blue light blinking ? [Y/n]")
    if (response == "n"):
        sys.exit("Check blue light cables")
    response = input("Is the green light blinking ? [Y/n]")
    if (response == "n"):
        sys.exit("Check green light cables")

        

    output, time_array = pam.read_detector()
    test_stop_measurement(link)

    import matplotlib.pyplot as plt
    plt.plot(output[arduino_green], label = "arduino_green")
    plt.plot(output[arduino_blue], label = "arduino_blue")
    plt.legend()
    plt.show()
