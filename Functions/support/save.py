from ..globalFunctions.utils import MyEncoder, applysigniftoall, get_size
import json
import numpy as np
import time

def SaveResults(input,cell,t,vsoma,traces,apcounts,aptimevectors,apinfo,totales,totalos,iOptogenx, succes_ratio,amps_SDeVstim,amps_SDoptogenx,pos_VTAeVstim,pos_VTAoptogenx,runtime,seed,results_dir):
    test_flag = input.test_flag
    # save input
    inputname = results_dir+'/input.json'
    inputData = {}
    inputData['seed'] = seed
    if 'eVstim' in input.stimopt.stim_type:
        inputData['totales'] = totales
    if 'Optogx' in input.stimopt.stim_type:
        inputData['totalos'] = totalos
    if input.signif is not None:
        inputData = applysigniftoall(inputData,input.signif)
    inputData['runtime'] = runtime
    inputData['settings'] = input.todict(reduce=True,inplace=False) # this line last because inplace=False does not work -> always inplace

    with open(inputname, 'w') as outfile:
        json.dump(inputData, outfile, indent = 4, signif=input.signif,  cls=MyEncoder)

    # save results
    resultsname = results_dir+'/data.json'
    data = {}
    data['APs'] = addAPinfotoResults(apcounts,aptimevectors,apinfo) if apcounts is not None else None
    data['t'] = np.array(t) if t is not None else None
    data['vsoma'] = np.array(vsoma) if vsoma is not None else None
    data['traces'] = addTracestoResults(traces,input.samplingFrequency/input.analysesopt['samplefrequency_traces']) if traces is not None else None
    data['Optogxstim'] = addOSinfo(cell, input.cellsopt['opsin_options']['opsinmech'],input.stimopt['Ostimparams'])
    data['Optogxstim']['iOptogenx'] = iOptogenx
    data['eVstim'] = addESinfo(cell,input.stimopt['Estimparams'])
    data['SDcurve']= {'eVstim': amps_SDeVstim, 'Optogenx': amps_SDoptogenx}
    data['VTApoints'] = {'eVstim': pos_VTAeVstim, 'Optogenx': pos_VTAoptogenx}
    data['succes_Ratio'] = succes_ratio
    datasize = get_size(data)
    if datasize>input.resultsmemlim:
        data['traces'] = f"excluded from saved file because mem limit {input.resultsmemlim} is exceeded {datasize} "
        print(data['traces'])
    if input.signif is not None:
        data = applysigniftoall(data,input.signif)


    noCorrectSave = True
    counter = 1
    while noCorrectSave:
        print(f'saving results, attempt {counter}')
        with open(resultsname, 'w') as outfile:
            json.dump(data, outfile, indent = 4, signif=input.signif,  cls=MyEncoder)
        if not test_flag:
            time.sleep(counter*10)
        try:
            #check if we can reload the saved file
            with open(resultsname, 'r') as testfile:
                intm = json.load(testfile)
            #succesfully reloaded file => end while
            noCorrectSave = False
        except ValueError:
            # if not succesfully reloaded try another time with longer pause but maximum 10 times
            if counter>10:
                noCorrectSave = False
            counter+=1
    print('\n!!Save successful !!')
    return inputData, data

def addOSinfo(cell,opsinmech,stiminfo):
    out = {}
    out['gbar_opsin'] = []
    out['intensity'] = []
    out['segname'] = []
    for sec in cell.allsec:
        for seg in sec:
            out['gbar_opsin'].append(getattr(seg,f"gchr2bar_{opsinmech}") if hasattr(seg,opsinmech) else np.nan)
            out['segname'].append(str(seg))
            out['intensity'].append(seg.os_xtra if hasattr(seg,'xtra') else np.nan)

    out['stim_info'] = stiminfo
    out['gbar_opsin'] = tuple(out['gbar_opsin'])
    out['segname'] = tuple(out['segname'])
    out['intensity'] = tuple(out['intensity'])
    return out
def addESinfo(cell,stiminfo):
    out = {}
    out['potential'] = []
    out['segname'] = []
    for sec in cell.allsec:
        for seg in sec:
            out['segname'].append(str(seg))
            out['potential'].append(seg.es_xtra if (hasattr(seg,'xtra') and hasattr(seg,'extracellular')) else np.nan)

    out['stim_info'] = stiminfo
    out['segname'] = tuple(out['segname'])
    out['potential'] = tuple(out['potential'])
    return out
def addAPinfotoResults(apcounts,aptimevectors,apinfo):
    out = {}
    out['apcounts'] = [x.n for x in apcounts]
    out['aptimes'] = [list(x) for x in aptimevectors]
    out['apinfo'] = apinfo
    return out
def addTracestoResults(traces,df):
    out = {}
    for k,v in traces.items():
        out[k] = {}
        for trace,name in zip(*v.values()):
            out[k][name] = np.array(trace)[::max(int(df),1)]
    return out
