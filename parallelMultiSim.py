#script that allows you to start multiple similuation in parallel independently of eachother
#Requires input from user, i.e., path to file that contains paths to input of simulation
import time
import os
import subprocess
import sys

Narg = len(sys.argv)

path_tostoreFinished = "."
numprocess = 2 #should not go higher than  num of CPUs
initpause = 300 #s
updatepause = 31 #s
verbose = True
gifsim = False
filepath = sys.argv[1]
if Narg>2:
    numprocess = int(sys.argv[2])
if Narg>3:
    initpause = int(sys.argv[3])
if Narg>4:
    updatepause = int(sys.argv[4])
if Narg>5:
    gifsim = bool(sys.argv[5])

#creata a empty file were finished simulations will be collected
finishedpath = os.path.join(path_tostoreFinished, 'finfished.txt')
with open(finishedpath, 'w') as fp:
    pass

# read all different inputs for simulations
with open(filepath) as f:
    inputsToDO = [line.rstrip('\n') for line in f]


# initialise flags to record process
busy_process_flags = [False]*numprocess
busy_processes = ['']*numprocess
process_init = [True]*numprocess

N = len(inputsToDO)

FLAG = 'True' #FLAG will be true as long as there are inputsToDO
while FLAG:

    #init processes in new console window if (virtual) process is not busy yet
    for i in range(numprocess):
        if FLAG and not busy_process_flags[i]:
            if i>0 & process_init[i]:
                # here we want to make sure that processes do not finish at same time ==> otherwise result will be overwritten
                time.sleep(initpause)
                process_init[i] = False
            busy_process_flags[i] = True
            myinput = inputsToDO.pop(0)
            if not gifsim:
                subprocess.Popen(["runmultiSim.bat",rf"{myinput}",rf"{finishedpath}"],creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(["gifsim\\runmultiSimgif.bat",rf"{myinput}",rf"{finishedpath}"],creationflags=subprocess.CREATE_NEW_CONSOLE)
            busy_processes[i] = myinput
            FLAG = len(inputsToDO)>0


    # wait before checking of process has finished
    time.sleep(updatepause)

    with open(os.path.join(finishedpath),'r') as f:
        finished = [line.rstrip('\n') for line in f]

    if verbose:
        perc = (1-len(inputsToDO)/N)*100
        print(f"\nfinished {perc:5.2f}%:\n", finished)
        print("\nBusy:", busy_processes)

    for ix,x in enumerate(busy_processes):
        if x in finished:
            busy_process_flags[ix] = False
print('finish')