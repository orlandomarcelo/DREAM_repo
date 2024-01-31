from subprocess import call
import time

def focus():
    print("FOCUS")
    #call(["python", "autofocus_wider.py"])
    call(["python", "autofocus_investigate.py"])

def dark_adaptation():
    print("DARK ADAPTATION")
    time.sleep(15*60)

def measurement_loop(n):
    print("MEASUREMENT LOOP")
    # MEASUREMENTS BEFORE ACTIVATION
    for i in range(n):
        call(["python", "qE_OJIP_calib.py"])
        
        
def assess_SP():

    print("assess_SP")
    # MEASUREMENTS BEFORE ACTIVATION
    call(["python", "qE_OJIP_test_SP.py"])
    
    
def activation_and_rest(activation_time_HL = 240,
                        level_HL = 450,
                        filter_HL = 1,
                        rest_time_LL=45, 
                        level_LL=150, 
                        filter_LL=2):
    # 2H ACTIVATION   
    print("ACTIVATION") 
    call(["python", "apply_constant_light.py", "with", "light_time=%d"%activation_time_HL, "limit_blue_high=%d"%level_HL, "actinic_filter=%d"%filter_HL])    
    print("RELAX LOW LIGHT")
    call(["python", "apply_constant_light.py", "with", "light_time=%d"%rest_time_LL,  "limit_blue_high=%d"%level_LL, "actinic_filter=%d"%filter_LL, "exposure=1000", "gain=100"])

def photoinhibition_relaxation():
    call(["python", "dark_relaxation.py"])

    
    
# before activation
#activation_and_rest(activation_time_HL=0)#level_HL = 250, activation_time_HL=60*3) #2H45 - 2H HL 45min LL

#focus()
#dark_adaptation() #15min
#assess_SP()
#measurement_loop(4) # total 2H, 15 min HL-15min dark en boucle

#activation
for i in range(5, 90): # 3x de suite 2H de haute lumière et entre chaque exposition regarder l'état des algues
#    focus()
    activation_and_rest(activation_time_HL=4*60)#level_HL = 250, activation_time_HL=60*3) #2H45 - 2H HL 45min LL
    focus()
    dark_adaptation() #15min
    assess_SP()
    measurement_loop(4) #total 2H, 15 min HL-15min dark en boucle
    
photoinhibition_relaxation()