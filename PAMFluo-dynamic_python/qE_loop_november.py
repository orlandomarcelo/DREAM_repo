from subprocess import call
import time

"""
for i in range(9):
    
    call(["python", "autofocus_investigate.py"])
    call(["python", "IBPC_loop.py"])
    time.sleep(4*3600)
    call(["python", "autofocus_investigate.py"])

    for i in range(4):
        call(["python", "qE_OJIP_calib.py"])
        
"""

for i in range(9):
    for i in range(5):
        call(["python", "autofocus_investigate.py"])
        for i in range(5):
            call(["python", "qE_OJIP_calib.py"])
    time.sleep(4*3600)
   