from serial import *
from python_comm import *
import numpy as np
import tkinter as tk

root = tk.Tk()
root.title('Command interface')
frame = tk.Frame(root)
frame.pack()

x = 0
y = 0
z = 0

port_arduino = 'COM6' #adapt the port to your arduino port !
link = Serial(port_arduino, 115200)

"""--------Useful functions-------------"""

#x displacement
def move_dx (link, dx, dt=-1) :
    if dt == -1:
        dt = dx*50 +1#adjust displacement speed according to the displacement length dx
    handle_move(link, np.abs(dt), dx, 0, 0) # send commmand to the Arduino
    global x
    global var_x
    x += dx
    var_x.set("x = "+str(x)) #allows to display the global x displacement

#x displacement
def move_dz (link, dz, dt=-1) :
    if dt == -1:
        dt = dz*50 +1#adjust displacement speed according to the displacement length dx
    handle_move(link, np.abs(dt), 0, 0, dz) # send commmand to the Arduino
    global z
    global var_z
    z += dz
    var_z.set("z = "+str(z)) #allows to display the global x displacement

# y displacement
def move_dy (link, dy, dt = -1) :
    if dt == -1:
        dt = dy//10 +1#adjust displacement speed according to the displacement length dy (considering the gearbox with ration x100)
    handle_move(link, np.abs(dt), 0, dy, 0) # send commmand to the Arduino
    global y
    global var_y
    y += dy//100
    var_y.set("y = "+ str(y)) #allows to display the global y displacement

def move_x_entry(): #move using user input value
    dx = np.int(entry_x.get())
    move_dx(link, dx, dt=-1)

def move_z_entry(): #move using user input value
    dz = np.int(entry_z.get())
    move_dz(link, dz, dt=-1)


def move_y_entry(): #move using user input value
    dy = np.int(entry_y.get())
    move_dy(link, dy*100, dt=-1)


def handle_enable(link, enable): #enable or diable to allow automatic or manual commmand respectively
    send_command(link, "E[%d]"%int(enable))
        
def toggle(): #Enable or disable motor control
    if enable_button.config('text')[-1] == 'Enable':
        handle_enable(link, 1)
        enable_button.config(text='Disable')
    else:
        handle_enable(link, 0)
        enable_button.config(text='Enable')


"""--------Buttons-----------"""
#leave the app
quit_button = tk.Button(frame, 
                   text="QUIT", 
                   fg="red",
                   command=quit)

quit_button.grid(column=0, row=2, ipadx=5, pady=5)

#enable or disable motors
enable_button = tk.Button(frame, text="ENABLE", fg="black", command=toggle)
enable_button.grid(column= 1, row=2, ipadx=5, pady=5)


#get global displacement values x and y
var_x = tk.StringVar()
label_x = tk.Label(frame, textvariable = var_x) #shows as text in the window
label_x.grid(column=0, row=0, columnspan=5, ipadx=5, pady=5)

var_y = tk.StringVar()
label_y = tk.Label(frame, textvariable = var_y) #shows as text in the window
label_y.grid(column=0, row=1, columnspan=5, ipadx=5, pady=5)

var_z = tk.StringVar()
label_z = tk.Label(frame, textvariable = var_z) #shows as text in the window
label_z.grid(column=0, row=2, columnspan=5, ipadx=5, pady=5)


# user inputs valus and clicks to move
entry_x = tk.Entry(frame)
entry_x.grid(column = 4, row=0, ipadx=5, pady=5 )    
user_x = tk.Button(frame, text = 'move x', command = move_x_entry)
user_x.grid(column = 5, row=0, ipadx=5, pady=5 )

entry_y = tk.Entry(frame)
entry_y.grid(column = 4, row=1, ipadx=5, pady=5 )    
user_y = tk.Button(frame, text = 'move y', command = move_y_entry)
user_y.grid(column = 5, row=1, ipadx=5, pady=5 )


entry_z = tk.Entry(frame)
entry_z.grid(column = 4, row=2, ipadx=5, pady=5 )    
user_z = tk.Button(frame, text = 'move z', command = move_z_entry)
user_z.grid(column = 5, row=2, ipadx=5, pady=5 )

# predefined movements x and y (+/- 100, 5à or 10)

################ X ##################

button2 = tk.Button(frame,
                   text="x - 100", fg="blue", bg="yellow",
                   command = lambda :move_dx(link, -100, -1))

button2.grid(column=0, row=3, ipadx=5, pady=5)

button3 = tk.Button(frame,
                   text="x - 50", fg="blue", bg="yellow",
                   command = lambda :move_dx(link, -50, -1))

button3.grid(column=0, row=4, ipadx=5, pady=5)

button4 = tk.Button(frame,
                   text="x - 10", fg="blue", bg="yellow",
                   command = lambda : move_dx(link, -10, -1))

button4.grid(column=0, row=5, ipadx=5, pady=5)

button5 = tk.Button(frame,
                   text="x + 10", fg="blue", bg="yellow",
                   command = lambda : move_dx(link, 10, -1))

button5.grid(column=1, row=3, ipadx=5, pady=5)

button6 = tk.Button(frame,
                   text="x + 50", fg="blue", bg="yellow",
                   command = lambda : move_dx(link, 50, -1))

button6.grid(column=1, row=4, ipadx=5, pady=5)

button7 = tk.Button(frame,
                   text="x + 100", fg="blue", bg="yellow",
                   command = lambda : move_dx(link, 100, -1))

button7.grid(column=1, row=5, ipadx=5, pady=5)


################ Y ##################


button8 = tk.Button(frame,
                   text="y - 100", fg="white", bg="red",
                   command = lambda : move_dy(link, -100*100, -1))

button8.grid(column=3, row=3, ipadx=5, pady=5)

button9 = tk.Button(frame,
                   text="y - 50", fg="white", bg="red",
                   command = lambda : move_dy(link, -50*100, -1))

button9.grid(column=3, row=4, ipadx=5, pady=5)

button10 = tk.Button(frame,
                   text="y - 10", fg="white", bg="red",
                   command = lambda : move_dy(link, -10*100, -1))

button10.grid(column=3, row=5, ipadx=5, pady=5)

button11 = tk.Button(frame,
                   text="y + 10", fg="white", bg="red",
                   command = lambda : move_dy(link, 10*100, -1))

button11.grid(column=4, row=3, ipadx=5, pady=5)

button12 = tk.Button(frame,
                   text="y + 50", fg="white", bg="red",
                   command = lambda : move_dy(link, 50*100, -1))
button12.grid(column=4, row=4, ipadx=5, pady=5)

button13 = tk.Button(frame,
                   text="y + 100", fg="white", bg="red",
                   command = lambda : move_dy(link, 100*100, -1))

button13.grid(column=4, row=5, ipadx=5, pady=5)

################ Z ##################
button14 = tk.Button(frame,
                   text="z - 100", fg="white", bg="blue",
                   command = lambda : move_dz(link, -100, -1))

button14.grid(column=5, row=3, ipadx=5, pady=5)

button15 = tk.Button(frame,
                   text="z - 50", fg="white", bg="blue",
                   command = lambda : move_dz(link, -50, -1))

button15.grid(column=5, row=4, ipadx=5, pady=5)

button16 = tk.Button(frame,
                   text="z - 10", fg="white", bg="blue",
                   command = lambda : move_dz(link, -10, -1))

button16.grid(column=5, row=5, ipadx=5, pady=5)

button17 = tk.Button(frame,
                   text="z + 10", fg="white", bg="blue",
                   command = lambda : move_dz(link, 10, -1))

button17.grid(column=6, row=3, ipadx=5, pady=5)

button18 = tk.Button(frame,
                   text="z + 50", fg="white", bg="blue",
                   command = lambda : move_dz(link, 50, -1))
button18.grid(column=6, row=4, ipadx=5, pady=5)

button19 = tk.Button(frame,
                   text="z + 100", fg="white", bg="blue",
                   command = lambda : move_dz(link, 100, -1))
button19.grid(column=6, row=5, ipadx=5, pady=5)



root.mainloop()