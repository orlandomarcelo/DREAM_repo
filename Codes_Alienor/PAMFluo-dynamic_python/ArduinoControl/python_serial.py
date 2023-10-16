import time
import json
import traceback

#Source ROMI Github: https://github.com/romi/romi-rover-build-and-test
def send_command(link, s):
    print("Command: %s" % s)
    command = "#" + s + ":xxxx\r\n"
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


    
def add_digital_pulse(link, pin, offset, period, duration, slave): #slave=0: indépendant, slave=1: esclave
    offset_s = offset//1000
    offset_ms = offset%1000
    period_s = period//1000
    period_ms = period%1000
    duration_s = duration//1000
    duration_ms = duration%1000
    send_command(link, "d[%d,%d,%d,%d,%d,%d,%d,%d]" % (pin, offset_s, offset_ms, period_s,period_ms, duration_s,
                                                       duration_ms, slave))

def add_master_digital_pulse(link, pin, offset, period, duration, slave): 
    offset_s = offset//1000
    offset_ms = offset%1000
    period_s = period//1000
    period_ms = period%1000
    duration_s = duration//1000
    duration_ms = duration%1000
    send_command(link, "m[%d,%d,%d,%d,%d,%d,%d,%d]" % (pin, offset_s, offset_ms, period_s,period_ms, duration_s,
                                                       duration_ms, slave))


def start_measurement(link):
    send_command(link, "b")

def stop_measurement(link):
    send_command(link, "e")
    reset_arduino(link)

#Source: https://forum.arduino.cc/index.php?topic=38981.0
def reset_arduino(link):
    link.setDTR(False) # Drop DTR
    time.sleep(0.022)    # 22ms is what the UI does.
    link.setDTR(True)  # UP the DTR back
    time.sleep(2)


