import serial #PySerial needs to be installed in your system
import time   # for Time 


# Generate file name using Current Date and Time

filepath = "C:/Users/Biophysique/Desktop/Light_data/ard"
current_local_time = time.localtime() #Get Current date time
filename           = time.strftime("%d_%B_%Y_%Hh_%Mm_%Ss",current_local_time)# 24hour clock format
filename           = filepath + filename + '_daq_log.csv'
print(f'Created Log File -> {filename}')


#Create a csv File header

with open(filename,'w+') as csvFile:
    csvFile.write('No;Time;Read\n')
    
log_count = 1

#COM 4 may change with system
SerialObj = serial.Serial('COM4',115200)
                
#Log continously to a file by querying the arduino                 
while 1:
    ReceivedString = SerialObj.readline()       # Change to receive  mode to get the data from arduino,Arduino sends \n to terminate
    ReceivedString = str(ReceivedString,'utf-8')# Convert bytes to string of encoding utf8
    if ReceivedString == "END":
        break
    
    valueslist = ReceivedString.split(';')  # Split the string into 4 values at '-'  
    #print(f'AN1={tempvalueslist[0]} AN2={tempvalueslist[1]} AN3={tempvalueslist[2]} AN4={tempvalueslist[3]}')
    
    seperator = ';'
    

    # create a string to write into the file
    if len(valueslist) == 2:
        
        log_file_text1 = str(log_count) + seperator 
        log_file_text2 = valueslist[0] + seperator + valueslist[1]
        log_file_text3 =  log_file_text1 +  log_file_text2
        
        # write to file .csv
        with open(filename,'a') as LogFileObj:
            LogFileObj.write(log_file_text3)
            
        #print(log_file_text3)
        log_count = log_count + 1 #increment no of logs taken
    

SerialObj.close()          # Close the port