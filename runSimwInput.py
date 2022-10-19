import sys
import os
import json
if __name__ == '__main__':

    from simulation import *
    import Functions.setup as stp

    try:
        filepath= sys.argv[1]
        if ',' in filepath:
            fps = filepath.split(',')
            filepath = fps[0]
            for i,val in enumerate(fps):
                try:
                    sys.argv[1+i] = val
                except IndexError:
                    sys.argv.append(val)
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
    #data = stp.simParams(myinput)
    if len(sys.argv)>3 and len(sys.argv)%2==0:
        gridFieldStimulation(myinput,**{sys.argv[i+2]:float(sys.argv[i+3]) for i in range(len(sys.argv)-2)[::2]})
    elif len(sys.argv)%2==1:
        raise ValueError('nr of input arguments should be odd')
    else:
        gridFieldStimulation(myinput)
