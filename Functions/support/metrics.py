import numpy as np
def calciOptogenx(input,t,traces):
    iOptogenx = None
    if input.analysesopt.recordTotalOptogeneticCurrent:
        if traces is None:
            print('try to recordTotalOptogeneticCurrent but not traces recorded (traces is None)')
            return iOptogenx
        iOptogenx = {'abs':{'total':0}, 'spec':{'total':0}}
        tintm = np.array(t)
        idx = (tintm>=input.stimopt.Ostimparams.delay) & (tintm<=(input.stimopt.Ostimparams.delay+input.stimopt.Ostimparams.dur+1000))
        for seg, icurrent in zip(traces['iOptogenx']['names'],traces['iOptogenx']['traces']):
            icurrent = np.array(icurrent)
            name = str(seg).split('.',1)[-1]
            iOptogenx['spec'][name] = np.trapz(icurrent[idx],x=tintm[idx])
            iOptogenx['abs'][name] = iOptogenx['spec'][name] * seg.area()
            iOptogenx['abs']['total'] += iOptogenx['abs'][name]
            iOptogenx['spec']['total'] += iOptogenx['spec'][name]
        del traces['iOptogenx']
    return iOptogenx

def calcSuccesratio(input,t,spikeTimes,ostim_time,ostim_amp,estim_time,estim_amp):
    succesRatio = None
    if input.analysesopt.recordSuccesRatio:
        succesRatio = {'succes':{},'spikeCountperstimRatio':{}}
        sRopt = input.analysesopt.succesRatioOptions
        tintm = np.array(t)
        sT = np.array(spikeTimes)
        sRopt['type'] = sRopt['type'] if isinstance(sRopt['type'],list) else [sRopt['type']]
        sRopt['window'] = sRopt['window'] if isinstance(sRopt['window'],list) else [sRopt['window']]; sRopt['window'] = int(len(sRopt['type'])/len(sRopt['window']))*sRopt['window']
        for sRtype,sRwindow in zip(sRopt['type'],sRopt['window']):
            if sRtype=='eVstim':
                stimtime = estim_time
                stimamp = estim_amp
                stimparams = input.stimopt.Estimparams
            elif sRtype=='Optogenx':
                stimtime = ostim_time
                stimamp = ostim_amp
                stimparams = input.stimopt.Ostimparams
            else:
                raise ValueError('input.analysesopt.succesRatioOptions["type"] can only be eVstim or Optogenx')

            if stimtime is None or len(stimtime) == 0:
                continue
            else:
                stimtime = np.array(stimtime)
                stimamp = np.array(stimamp)

            stimamp = np.abs(stimamp)
            idx_on = (stimamp>0) & (np.roll(stimamp,1)<=0)
            idx_off = (stimamp>0) & (np.roll(stimamp,-1)<=0)
            t_on = stimtime[idx_on]
            t_off = stimtime[idx_off]
            if len(t_on)>len(t_off):
                # if stimulation end of simulaton time t_off could miss a final time point
                t_off = np.append(t_off,tintm[-1])
            t_off_spd = np.minimum(t_off+sRwindow,np.append(t_on[1:],t[-1])) # window for spike detection should always be smaller than start of next pulse
            intm = (sT>=t_on[:,None]) & (sT<t_off_spd[:,None])
            succesRatio['succes'][sRtype] = np.sum(np.any(intm,1))/len(t_on)
            succesRatio['spikeCountperstimRatio'][sRtype] = np.sum(intm)/len(t_on)


    return succesRatio
