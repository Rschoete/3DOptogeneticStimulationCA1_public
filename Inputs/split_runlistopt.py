#First run cliprunlistopt.py to get to do list
import numpy as np
folder = './Inputs/SDC20220929_invivoCA1pyrs5/'
inputsToDO = folder+'inputsToDO.csv'
ypossdefault = list(np.arange(0,2000,200)) # check simulation.gridFieldStimulation

splittype = 'ysplit'
delimiter = ','


with open(inputsToDO,'r') as f:
    inputsToDO = [line.rstrip('\n') for line in f]

columnheader = inputsToDO.pop(0) +f'{delimiter}key1' +f'{delimiter}yposs'+f'{delimiter}key2'+f'{delimiter}overWriteSave'



with open(folder+f'inputsToDO_{splittype}.csv','w') as file:
    file.write(columnheader+'\n')
    for  item in inputsToDO:
        for ypos in ypossdefault:
            file.write("%s\n" % delimiter.join([item,'yposs',str(ypos),'overWriteSave','0']))

