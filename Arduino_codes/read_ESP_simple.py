import serial
import csv

# Set the serial port and baud rate
ser = serial.Serial('COM4', 115200) # Replace 'COM3' with the serial port of your ESP32 board
ser.flushInput()

filepath = "C:/Users/Biophysique/Desktop/Light_data/"

# Create a CSV file and write the header row
with open(filepath + 'data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

# Read serial data and save it to the CSV file
while True:
    try:
        # Read a line of serial data
        line = ser.readline().decode('utf-8').rstrip()
        
        # Split the line into time and data values
        values = line.split(';')
        if len(values) == 2:
            time = values[0]
            data = values[1]
            
            # Append the time and data values to the CSV file
            with open('data.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([time, data])
                
    except KeyboardInterrupt:
        print('Keyboard interrupt')
        break
        
    except:
        print('Error occurred')
        continue
