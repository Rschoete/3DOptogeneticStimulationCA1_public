from .globalFunctions.utils import MyEncoder, applysigniftoall, get_size
from .globalFunctions.ExtracellularField import _singleSlicePlot
import Functions.globalFunctions.morphology_v2 as mphv2
import matplotlib.pyplot as plt
import json
import numpy as np
import time

colorkeyval = {'soma':'tab:red', 'axon':'tomato','apical trunk':'tab:blue','apical trunk ext':'royalblue', 'apical tuft': 'tab:green','apical obliques': 'tab:cyan', 'basal dendrites': 'tab:olive', 'unclassified':[0,0,0]}

def AnalysesWrapper(h,input,cell,t,vsoma,traces,ostim_time,ostim_amp,estim_time,estim_amp,aptimevectors,apinfo,idx_sR,amps_SDeVstim,amps_SDoptogenx,pos_VTAeVstim,pos_VTAoptogenx,fig_dir):


    iOptogenx = None; VTAOptogenx = None; VTAeVstim = None

    # create colored section plot
    aopt = input.analysesopt
    if aopt.sec_plot_flag:
        print("\t* Section plot")
        ax = cell.sec_plot()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=90, azim=-90)
        if input.save_flag:
            savename = f"{fig_dir}/sec_plot.png"
            plt.savefig(savename)

    # print section positions
    if aopt.print_secpos:
        print("\t* Section positons")
        cell.gather_secpos(print_flag = True)

    # create shapePlots
    print("\t* Shape plots")
    shapePlot(h,cell, aopt.shapeplots, aopt.shapeplot_axsettings, input.save_flag and aopt.save_shapeplots,figdir=fig_dir,extension=aopt.shapeplots_extension)

    # recordTotalOptogeneticCurrent
    iOptogenx = calciOptogenx(input,t,traces)

    # succes Ratio
    succes_ratio = calcSuccesratio(input,t,aptimevectors[idx_sR],ostim_time,ostim_amp,estim_time,estim_amp)

    # plot recorded traces
    if vsoma is not None:
        print("\t* Plot traces")
        plot_traces(t,vsoma,traces,ostim_time,ostim_amp,estim_time,estim_amp, aopt.tracesplot_axsettings, input.save_flag and aopt.save_traces, figdir=fig_dir,extension=aopt.traces_extension)
    elif not "normal" in input.simulationType:
        print('\t* no traces to plot because "normal" not in input.stimulationType')
    else:
        print("\t* no traces to plot, normally always should plot vsoma???")

    # rastergram
    if aptimevectors is not None:
        print("\t* Raster plot")
        rasterplot(cell,aptimevectors,apinfo,np.array(t),input.save_flag and aopt.save_rasterplot, figdir = fig_dir, **aopt.rasterplotopt)
    elif not "normal" in input.simulationType:
        print('\t* no raster plot because "normal" not in input.stimulationType')

    # SDcurve plots
    if amps_SDeVstim is not None:
        SDcopt = input.analysesopt.SDeVstim
        try:
            SDcurveplot(SDcopt,amps_SDeVstim,'V_e stim amp [V]','eVstim',input.save_flag and aopt.save_SDplot, figdir = fig_dir)
        except Exception as E:
            print(E)


    if amps_SDoptogenx is not None:
        SDcopt = input.analysesopt.SDOptogenx
        try:
            SDcurveplot(SDcopt,amps_SDoptogenx,'Light Intensity stim amp [W/m2]','optogenx',input.save_flag and aopt.save_SDplot, figdir = fig_dir)
        except Exception as E:
            print(E)

    if pos_VTAeVstim is not None:
        VTAopt = input.analysesopt.VTAeVstim
        try:
            VTAeVstim = VTAplot(VTAopt,pos_VTAeVstim,input.stimopt.Estimparams,'eVstim',input.save_flag and aopt.save_VTAplot, figdir = fig_dir)
        except Exception as E:
            print(E)
    if pos_VTAoptogenx is not None:
        VTAopt = input.analysesopt.VTAOptogenx
        try:
            VTAOptogenx = VTAplot(VTAopt,pos_VTAoptogenx,input.stimopt.Ostimparams,'optogenx',input.save_flag and aopt.save_VTAplot, figdir = fig_dir)
        except Exception as E:
            print(E)

    if input.plot_flag:
        plt.show(block=False)

    return iOptogenx, succes_ratio, VTAOptogenx, VTAeVstim

def calciOptogenx(input,t,traces):
    iOptogenx = None
    if input.analysesopt.recordTotalOptogeneticCurrent:
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

def shapePlot(h,cell,shapeplotsinfo, axsettings,save_flag, figdir = '.', extension = '.png'):
    if len(shapeplotsinfo)>0:
        if isinstance(axsettings,dict):
            axsettings = len(shapeplotsinfo)*[axsettings]
        elif len(axsettings)==1 and isinstance(axsettings,list):
            axsettings = len(shapeplotsinfo)*axsettings

        for pi,axsetting in zip(shapeplotsinfo,axsettings):
            fig = plt.figure(figsize = axsetting.get('figsize',(16,10)))
            ax = plt.subplot(111,projection='3d')
            mphv2.shapeplot(h,ax,**pi)
            ax.set_zlim(axsetting.get('zlim',[-300,300]))
            ax.set_xlim(axsetting.get('xlim',[-300,300]))
            ax.set_ylim(axsetting.get('ylim',[-200,400]))
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.view_init(elev=axsetting.get('elev',90), azim=axsetting.get('azim',-90))
            ax.set_title(f"{cell.templatename} / {cell.morphology} / {cell.celltype}")
            if save_flag:
                savename = f"{figdir}/shapeplot_{pi['cvals_type']}.{extension}"
                fig.savefig(savename)

def plot_traces(t,vsoma,traces,ostim_time,ostim_amp,estim_time,estim_amp,figsettings,save_flag,figdir = '.', extension = '.png'):
    t = np.array(t)
    vsoma = np.array(vsoma)
    # plot traces
    fig,ax = plt.subplots(1,1,tight_layout=True,figsize=(9,6))
    ax.plot(t,vsoma)

    # fill locations of pulses
    for stimtime,stimamp,clr in zip([ostim_time,estim_time],[ostim_amp,estim_amp],['tab:blue','gold']):
        stimamp = np.array(stimamp)
        if len(stimamp>0):
            Iopt_np = np.array(stimamp)
            idx_on = (Iopt_np>0) & (np.roll(Iopt_np,1)<=0)
            idx_off = (Iopt_np>0) & (np.roll(Iopt_np,-1)<=0)
            t_np = np.array(stimtime)
            t_on = t_np[idx_on]
            t_off = t_np[idx_off]
            if len(t_on)>len(t_off):
                # if illumination till end of simulaton time t_off could miss a final time point
                t_off = np.append(t_off,t_np[-1])
            for ton,toff in zip(t_on,t_off):
                ax.axvspan(ton,toff,color=clr,alpha=0.2)

    #set labels
    ax.set_xlim([0,t[-1]])
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('V [mV]')
    ax.set_title('soma')
    if save_flag:
        savename = f"{figdir}/traces_vsoma.{extension}"
        fig.savefig(savename)

    # plot all recorded traces
    for k,v in traces.items():

        nrsubplots = int(np.ceil(len(v['traces'])/4))
        nrfigs = int(np.ceil(nrsubplots/3))

        sp_count = 0
        for ifig in range(nrfigs):
            fig = plt.figure(tight_layout=True)
            if sp_count+3<nrsubplots:
                axs = fig.subplots(3,1,sharex=True)
                sp_count+=3
            else:
                axs = fig.subplots(nrsubplots-sp_count,1,sharex=True)
                sp_count+=nrsubplots-sp_count
                if not isinstance(axs,np.ndarray):
                    axs = [axs]

            for i,ax in enumerate(axs):
                t1 = ifig*12+i*4
                t2 = ifig*12+(i+1)*4
                for trace, name in zip(v['traces'][t1:t2],v['names'][t1:t2]):
                    ax.plot(t,trace,label=name.split('.',1)[-1])
                ax.legend()
            fig.suptitle(k)

            if save_flag:
                savename = f"{figdir}/traces_{k}{ifig+1}.{extension}"
                fig.savefig(savename)

def setup_recordTraces(h,recordTraces,cell,Dt):

    t = h.Vector().record(h._ref_t,Dt)
    vsoma = h.Vector().record(cell.soma[0](0.5)._ref_v,Dt)

    traces = {}

    for k,v in recordTraces.items():
        traces[k] = {'traces':[], 'names':[]}

        if isinstance(v['sec'],str):
            v['sec'] = convert_strtoseclist(cell,v['sec'])
        elif all([type(x)==str for x in v['sec']]):
            v['sec'] = [convert_strtosec(cell,x) for x in v['sec']]
        else:
            raise ValueError('sec should either be list or str')

        if isinstance(v['loc'],(float,int)):
            v['loc'] = len(v['sec'])*[v['loc']]
        elif len(v['loc']) != len(v['sec']):
            raise ValueError('loc and sec should have same length')

        if isinstance(v['var'],str):
            v['var'] = len(v['sec'])*[v['var']]
        elif len(v['var']) != len(v['sec']):
            raise ValueError('var and sec should have same length')

        if 'mech' in v.keys():
            if isinstance(v['mech'],str):
                v['mech'] = len(v['sec'])*['_'+v['mech']]
            elif len(v['mech']) != len(v['sec']):
                raise ValueError('mech and sec should have same length')
        else:
            v['mech'] = len(v['sec'])*['']

        for sec,loc,var,mech in zip(*v.values()):
            if hasattr(sec(loc),f'_ref_{var}{mech}'):
                name = f'{str(sec(loc))}_{k}'
                traces[k]['traces'].append(h.Vector().record(getattr(sec(loc),f'_ref_{var}{mech}'),Dt))
                traces[k]['names'].append(name)

    return t, vsoma, traces

def rasterplot(cell,aptimevectors,apinfo,t,save_flag, figdir, markersize = 2, marker = '.', xlims = None, colorkeyval=colorkeyval, extension='.png'):
    fig,ax =plt.subplots(1,1,tight_layout=True)
    for i, (aptv, clr) in enumerate(zip(aptimevectors,apinfo['colorlist'])):
        ax.scatter(list(aptv),i*np.ones(len(aptv)),s=markersize,c=clr,marker = marker)

    if xlims is None:
        xlims = [t[0], t[-1]]
    ax.set_xlim(xlims)
    ax.set_ylim([-1,i+1])
    ax.invert_yaxis()
    ax.set_xlabel('time [ms]')
    for k,v in colorkeyval.items():
        ax.plot(np.nan,np.nan,label=k,color=v)
    ax.legend(frameon=False)
    ax.set_title(f"{cell.templatename} / {cell.morphology} / {cell.celltype}")
    if save_flag:
        savename = f"{figdir}/rasterplot.{extension}"
        fig.savefig(savename)

def setup_recordAPs(h,recordAPs,cell, threshold, succesRatio_seg, preorder = False, colorkeyval=colorkeyval):
    apcounts = []
    aptimevectors = []
    segnames = []
    colorlist = []
    idx_sR = None
    if recordAPs == 'all':
        if preorder:
            seglist = [seg for sec in mphv2.allsec_preorder(h) for seg in sec]
        else:
            seglist = [seg for sec in cell.allsec for seg in sec]
    elif recordAPs == 'all0.5':
        if preorder:
            seglist = [sec(0.5) for sec in mphv2.allsec_preorder(h)]
        else:
            seglist = [sec(0.5) for sec in cell.allsec]
    elif (recordAPs is None) and (succesRatio_seg is not None):
        seglist = [convert_strtoseg(succesRatio_seg)]
        idx_sR = 0
    else:
        raise ValueError('recordAPs: can only be "all" or "all0.5"')
    for seg in seglist:
        segnames.append(str(seg).split('.',1)[-1])
        if segnames[-1] == succesRatio_seg:
            idx_sR = len(segnames)-1
        timevector = h.Vector()
        apc = h.APCount(seg)
        apc.thresh = threshold
        apc.record(timevector)
        apcounts.append(apc)
        aptimevectors.append(timevector)

        #store coloers for raster plot
        sec = seg.sec
        if 'soma' in str(sec):
            colorlist.append(colorkeyval['soma'])
        elif 'axon' in str(sec):
            colorlist.append(colorkeyval['axon'])
        elif sec in cell.apicalTrunk:
            colorlist.append(colorkeyval['apical trunk'])
        elif sec in cell.apicalTrunk_ext:
            colorlist.append(colorkeyval['apical trunk ext'])
        elif sec in cell.apicalTuft:
            colorlist.append(colorkeyval['apical tuft'])
        elif sec in cell.apical_obliques:
            colorlist.append(colorkeyval['apical obliques'])
        elif 'dend' in str(sec):
            colorlist.append(colorkeyval['basal dendrites'])
        else:
            colorlist.append([0,0,0])
    apinfo = {}
    apinfo['segnames'] = tuple(segnames)
    apinfo['colorlist']=tuple(colorlist)
    return tuple(apcounts), tuple(aptimevectors), apinfo, idx_sR

def SDcurveplot(SDcopt,amps,ylabel,savename,save_flag, figdir,extension='png'):
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(SDcopt.durs,np.abs([x if x is not None else np.nan for x in amps]))
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Duration [ms]')
        if save_flag:
            savename = f"{figdir}/SDplot_{savename}.{extension}"
            fig.savefig(savename)

def VTAplot(VTAopt,positions,stimparams,savename,save_flag,figdir,extension='png'):
    # assumes axial symmetry
    pos = [x for x in positions if x is not None]
    if len(pos)==0:
        return None
    pos = np.array(pos)
    pos0 = np.array(VTAopt.startpos)
    rpos = np.array([np.linalg.norm(pos[:,0:1],axis=1),pos[:,-1]]).T
    r0pos = np.array([np.linalg.norm(pos0[:,0:1],axis=1),pos0[:,-1]]).T

    fig = plt.figure()
    ax = plt.subplot(111)

    field = stimparams.field
    if len(stimparams.field)==3:
        if stimparams.structured:
            ax.pcolormesh(field[0],field[1],np.log10(field[2]).T,shading='auto')
        else:
            raise NotImplementedError
    else:
        if stimparams.structured:
            ax.pcolormesh(field[0],field[2],np.log10(np.squeeze(field[3][:,field[1]==0,:])).T,shading='auto')
        else:
            raise NotImplementedError


    for r0,r in zip(r0pos,rpos):
        ax.plot([r0[0],r[0]],[r0[-1],r[-1]],'X')
    ax.plot(rpos[:,0],rpos[:,1],'r')
    ax.set_ylabel('axial [mm]')
    ax.set_xlabel('radial [mm]')
    ymaxlim = 1.1*max(max(r0pos[:,1]),max(rpos[:,1]))
    xmaxlim = 1.1*max(max(r0pos[:,0]),max(rpos[:,0]))
    ax.set_ylim(max(ax.get_ylim()[0],-ymaxlim),ymaxlim)
    ax.set_xlim(max(ax.get_xlim()[0],-xmaxlim),xmaxlim)
    ax.invert_yaxis()
    if save_flag:
        savename = f"{figdir}/VTAplot_{savename}.{extension}"
        fig.savefig(savename)
    from scipy import integrate
    return {'VTA':integrate.simps(np.pi*rpos[:,0]**2,rpos[:,1]),'pos':positions}

def convert_strtoseclist(cell,location):
    if location.lower()=='soma':
        location = cell.soma
    elif location.lower()=='axon':
        location = cell.axon
    elif location.lower()=='all':
        location = cell.allsec
    elif location.lower()=='alldend':
        location = cell.alldend
    elif location.lower() in ['apic','apical']:
        location = cell.apical
    elif location.lower() in ['basal','dend']:
        location = cell.dend
    elif location.lower() == 'apicaltrunk':
        location = cell.apicalTrunk
    elif location.lower() == 'apicaltrunk_ext':
        location = cell.apicalTrunk_ext
    elif location.lower() == 'apicaltuft':
        location = cell.apicalTuft
    elif location.lower() == 'obliques':
        location = cell.apical_obliques
    elif location.lower() == 'apicalnotuft':
        location = cell.apicalTrunk_ext+cell.apical_obliques
    else:
        try:
            location = getattr(cell,location)
        except:
            raise ValueError(location)
    return location

def convert_strtoseg(cell,mystr):
    str_split = mystr.split('(',1)
    sec = str_split[0]

    secnr = int(sec.split('[',1)[-1][:-1])
    sec = sec.split('[',1)[0]
    loc = float(str_split[-1][:-1]) # :-1 to exclude closing bracket
    return getattr(cell,sec)[secnr](loc)

def convert_strtosec(cell,mystr):
    str_split = mystr.split('[',1)
    sec = str_split[0]
    if len(str_split)>1:
        secnr = float(str_split[-1][:-1])
        return getattr(cell,sec)[secnr]
    else:
        return getattr(cell,sec)

def get_dts_simdurs(simdur,dt_init,edts,esimtimes,odts,osimtimes):
    '''
    #test case succesful
    odts = [25,5,25,5,25,5,25]
    osimtimes = [100,101,110,111,120,121,130]
    edts = [25,4,20,4,20,4,25]
    esimtimes = [103,104,111,112,120.5,121.5,130]
    simdur = 135
    dt_init = 19
    '''
    if len(osimtimes)==0 and len(esimtimes)==0:
        dts = [dt_init]; simdurs = [simdur]
        return dts, simdurs
    elif len(osimtimes)==0:
        simtimes = esimtimes
        xdts = edts
    elif len(esimtimes)==0:
        simtimes = osimtimes
        xdts = odts

    else:
        xdts = np.array(odts)
        simtimes = np.array(osimtimes)

        for edt,est0,est1 in zip(edts,[0]+esimtimes[:-1],esimtimes):
            idx_l = simtimes<=est0
            idx = (simtimes>est0) & (simtimes<est1)
            idx_h = simtimes>=est1
            idx_sh = simtimes>est1
            dts_intm = np.concatenate((xdts[idx_l],np.minimum(np.array(xdts)[idx],edt)))
            dts_intm = np.append(dts_intm,min(xdts[idx_h][0],edt))
            xdts = np.concatenate((dts_intm,xdts[idx_sh]))
            simtimes = np.concatenate((simtimes[idx_l],simtimes[idx],np.array([est1]),simtimes[idx_sh]))
        xdts = list(xdts)
        simtimes = list(simtimes)

    # append simdur
    if simtimes[-1]<simdur:
        dts = list(np.minimum(xdts,dt_init))+[dt_init]
        simdurs = simtimes+[simdur]
    else:
        simtimes = np.array(simtimes)
        xdts = np.array(xdts)
        idx_l = simtimes<simdur
        idx_h= simtimes>=simdur
        dts=np.minimum(np.array(xdts)[idx_l],dt_init)
        simdurs = simtimes[idx_l]
        dts = np.append(dts,min(xdts[idx_h][0],dt_init))
        simdurs = np.append(simdurs,simdur)

    return dts,simdurs

