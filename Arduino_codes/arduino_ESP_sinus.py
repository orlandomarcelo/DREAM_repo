import serial
import math
import time

# PWM definitions

resolution = 16  # bits
max_amp = 2 ** 16

# Define the frequency and amplitude of the sinus function
freq = 1  # in Hzpytho
amp = max_amp/2  # max PWM value is 255, so set amplitude to half
refresh_rate = 1000  # LED refresh rate in Hz

# Initialize the serial connection to the Arduino
ser = serial.Serial('COM4', 9600)  # Change 'COM3' to the serial port of your Arduino


# Define the sinus function to generate the PWM value for the LED
def sin_wave(t):
    return int(amp * math.sin(2 * math.pi * freq * t) + amp)

# Loop forever, sending PWM values to the Arduino to control the LED
t = 0
while True:
    pwm_val = sin_wave(t)
    ser.write(bytes(str(pwm_val) + '\n', 'utf-16'))  # Send the PWM value to the Arduino
    t += 1 / refresh_rate  # Increment time by 1/act_rate of a second (refresh rate)
    time.sleep(1 / refresh_rate)  # Wait for 1/act_rate of a second before sending the next PWM value
