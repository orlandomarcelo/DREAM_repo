import serial
import math
import time

# Define the frequency and amplitude of the sinus function
freq = 1  # in Hz
amp = 127  # max PWM value is 255, so set amplitude to half
act_rate = 120 # 10 Hz


# Initialize the serial connection to the Arduino
ser = serial.Serial('COM3', 9600)  # Change 'COM3' to the serial port of your Arduino

# Define the sinus function to generate the PWM value for the LED
def sin_wave(t):
    return int(amp * math.sin(2 * math.pi * freq * t) + amp)

# Loop forever, sending PWM values to the Arduino to control the LED
t = 0
while True:
    pwm_val = sin_wave(t)
    ser.write(bytes(str(pwm_val) + '\n', 'utf-8'))  # Send the PWM value to the Arduino
    t += 1/act_rate  # Increment time by 1/60 of a second (60 Hz refresh rate)
    time.sleep(1/act_rate)  # Wait for 1/60 of a second before sending the next PWM value
