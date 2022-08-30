import sys
import os
import json
if __name__ == '__main__':

    from simulation import *
    import Functions.setup as stp

    try:
        filepath= sys.argv[1]
    except:
        filepath = 'test'
    print(filepath)
    if not filepath.endswith('.json'):
        filepath = filepath+'.json'
    if not filepath.startswith('./'):
        filepath_intm = './'+filepath
        if os.path.isfile(filepath_intm):
            filepath = filepath_intm
        else:
            filepath = './Inputs/'+filepath
    print(filepath)
    with open(filepath) as f:
        myinput = json.load(f)
    print(myinput)
    data = stp.simParams(myinput)
    gridFieldStimulation(data)
