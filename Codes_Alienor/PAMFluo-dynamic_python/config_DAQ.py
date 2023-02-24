

detector = ["intensity", "fluorescence", "arduino_blue", "arduino_green", "arduino_purple"]

intensity = 0
fluorescence = 1
arduino_blue = 2
arduino_green = 3
arduino_purple = 4

LED_red = 1
LED_blue = 2
LED_green = 3
LED_purple = 4


NI_blue = 0
NI_purple = 1
NI_green = 2

generator_analog = {NI_blue: "ao0",
                        NI_purple: "ao1",
                        NI_green: "ao2", 
}

generator_digital = {'trigger': "port0/line3"}

detector_analog = { intensity:"ai0", 
                    fluorescence:"ai1"}



detector_digital = {arduino_blue:"port0/line5", 
                    arduino_purple: "port0/line4", 
                    arduino_green : "port0/line0"}


trigger = { 'blue': '/Dev1/PFI8',  
            'purple': '/Dev1/PFI9', 
            'green': '/Dev1/PFI14', 
            'no_LED_trig': '/Dev1/PFI15'}


pins = {'blue': 10,
        'green': 12, 
        'purple': 11,
        'red': 9,
        'relay':8 ,
        'no_LED_trig': 7}


filters = {1:1,
                2:2,
                3:3,
                0.5:4,
                0:6}
sec = 1000
minute = 60*sec

save_figure_folder = "G:/DREAM/from_github/PAMFluo/Figures/"

