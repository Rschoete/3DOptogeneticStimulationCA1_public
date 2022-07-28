import os
import sys
import time
import json
import random
import numpy as np
from neuron import h
if os.path.exists("./Model/Mods/x86_64/libnrnmech.so"):
    # linux compiled file (on hpc)
    h.nrn_load_dll("./Model/Mods/x86_64/libnrnmech.so")
    print("succes load libnrnmech.so")
else:
    # above file should not exist locally -> load windows compiled nrnmech
    h.nrn_load_dll("./Model/Mods/nrnmech.dll")
    print("succes load nrnmech.dll")
from datetime import datetime
from Model import Cells
from matplotlib import use as mpluse
import Functions.globalFunctions.morphology_v2 as mphv2
import Functions.globalFunctions.ExtracellularField as eF
from Functions.globalFunctions.utils import MyEncoder, applysigniftoall

colorkeyval = {'soma':'tab:red', 'axon':'tomato','apical trunk':'tab:blue','apical trunk ext':'royalblue', 'apical tuft': 'tab:green','apical obliques': 'tab:cyan', 'basal dendrites': 'tab:olive', 'unclassified':[0,0,0]}

def optogeneticStimulation(input, verbose = False):

    test_flag = input.test_flag
    if not input.plot_flag:
        # With this backend plots are not shown but can be saved
        mpluse("AGG")
    else:
        mpluse("TKAGG")

    # important for reproducibility, seed based on ceratain date
    seed = input.seed
    rng = np.random.seed(int(seed))
    random.seed(int(seed))

    # Cell setup
    # ----------------------------------------------------------
    ##Load cell
    if input.cellsopt.neurontemplate in Cells.NeuronTemplates:
        cell = getattr(Cells,input.cellsopt.neurontemplate)(**input.cellsopt.init_options)
        if verbose:
            print(f'{input.cellsopt.neurontemplate} loaded')
    else:
        raise ValueError(f"input.neuronInput = {input.cellsopt.neurontemplate} is invalid. Possible templates are  {Cells.NeuronTemplates}")


    ## Plugin opsin
    oopt = input.cellsopt.opsin_options
    if oopt.opsinlocations is not None:
        if isinstance(oopt.opsinlocations,str):
            # if opsin location is a string convert to list of sections (method of neurontemplate class)
            oopt.opsinlocations = convert_strtoseclist(cell,oopt.opsinlocations)

        cell.insertOptogenetics(seclist = oopt.opsinlocations, opsinmech = oopt.opsinmech, set_pointer_xtra =  oopt.set_pointer_xtra)

        if oopt.distribution_source_hdistance is not None:
            oopt.distribution_source_hdistance = convert_strtoseg(oopt.distribution_source_hdistance)
        seglist, values = cell.distribute_mechvalue(oopt.distribution,method=oopt.distribution_method,source_hdistance = oopt.distribution_source_hdistance)
        cell.updateMechValue(seglist,values,'gchr2bar_'+oopt.opsinmech)


    ## Plugin extracellular
    if input.cellsopt.extracellular:
        cell.insertExtracellular()


    ## Cell transformation
    # Rotate cell
    ctopt = input.cellsopt.cellTrans_options
    if ctopt.rotate_flag:
        cell.rotate_Cell(ctopt.phi,ctopt.theta)
    # Translate cell
    if ctopt.move_flag:
        cell.move_Cell(ctopt.rt)
    del ctopt

    cell.updateXtraCoors() # important to update Xtra values after transformation. values stored in xtra mech are used to calculated received Ve and I


    ## check if all pointers set
    cell.check_pointers(input.autosetPointer) # autoset missing pointers to track a non change variable that equals 0 at init -> reason why sections do not receive light when no opsin present (in shape plots)


    # Stimulation setup
    # ----------------------------------------------------------
    ## Load fields
    # Potential field
    estim_amp = []; estim_time = []
    if 'eVstim' in input.stimopt.stim_type:
        params = input.stimopt.Estimparams
        if params['field'] is None:
            field = np.genfromtxt(params['filepath'],comments='%')
            params['field'] = eF.prepareDataforInterp(field, params['method_prepareData'])
            del field
        t = np.arange(0,input.duration*1.1, input.dt/10)
        estim_time,estim_amp,totales = eF.setXstim(cell.allsec, t, params['delay'], params['dur'], params['amp'], params['field'],params['structured'],params['pulseType'],stimtype='electrical',netpyne=False, **params['options'])
        if estim_amp is not None and len(estim_amp)>0:
            estim_amp = h.Vector(estim_amp)
            estim_time = h.Vector(estim_time)
            estim_amp.play(h._ref_estim_xtra, estim_time, True) #True -> interpolate

    # Optical field
    ostim_amp = []; ostim_time = []
    if 'Optogxstim' in input.stimopt.stim_type:
        params = input.stimopt.Ostimparams
        if params['field'] is None:
            field = np.genfromtxt(params['filepath'],comments='%')
            params['field'] = eF.prepareDataforInterp(field, params['method_prepareData'])
            del field
        t = np.arange(0,input.duration*1.1, input.dt/10)
        ostim_time,ostim_amp,totalos = eF.setXstim(cell.allsec, t, params['delay'], params['dur'], params['amp'], params['field'],params['structured'],params['pulseType'],stimtype = 'optical',netpyne=False, **params['options'])
        if ostim_amp is not None and len(ostim_amp)>0:
            # need to do at this level (does not work when in do not know why)
            ostim_amp = h.Vector(ostim_amp)
            ostim_time = h.Vector(ostim_time)
            ostim_amp.play(h._ref_ostim_xtra, ostim_time, True) #True -> interpolate



    # Analyses setup
    # ----------------------------------------------------------
    ## setup recording traces
    Dt = 1/input.samplingFrequency*1e3
    t, vsoma, traces = setup_recordTraces(input.analysesopt.recordTraces,cell,Dt)
    apcounts, aptimevectors, apinfo = setup_recordAPs(h,input.analysesopt.recordAPs,cell, threshold=input.analysesopt.apthresh, preorder= input.analysesopt.preordersecforplot)


    # Simulate
    # ----------------------------------------------------------
    h.dt = input.dt
    h.celsius = input.celsius
    h.finitialize(input.v0)
    h.continuerun(input.duration)



    # Analyse
    # ----------------------------------------------------------
    # Create
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M")
    results_dir = "./Results%s/Results_%s_%s_%s"%(input.resultsFolder, input.subfolderSuffix,input.cellsopt.neurontemplate,dt_string)
    fig_dir = results_dir+"/Figures"
    if input.save_flag:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)
    # create colored section plot
    aopt = input.analysesopt
    if aopt.sec_plot_flag:
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
        cell.gather_secpos(print_flag = True)

    # create shapePlots
    shapePlot(cell, aopt.shapeplots, aopt.shapeplot_axsettings, input.save_flag and aopt.save_shapeplots,figdir=fig_dir,extension=aopt.shapeplots_extension)

    # plot recorded traces
    plot_traces(t,vsoma,traces,ostim_time,ostim_amp,estim_time,estim_amp, aopt.tracesplot_axsettings, input.save_flag and aopt.save_traces, figdir=fig_dir,extension=aopt.traces_extension)

    # rastergram
    rasterplot(cell,aptimevectors,apinfo,np.array(t),input.save_flag and aopt.save_rasterplot, figdir = fig_dir, **aopt.rasterplotopt)

    if input.plot_flag:
        plt.show(block=False)

    print('finished')

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

    inputData['settings'] = input.todict(reduce=True,inplace=False) # this line last because inplace=False does not work -> always inplace

    with open(inputname, 'w') as outfile:
        json.dump(inputData, outfile, indent = 4, signif=input.signif,  cls=MyEncoder)

    # save results
    resultsname = results_dir+'/data.json'
    data = {}

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
    print('save successful')
    # [ ]: feature extraction at least Firing rate, save in way known for all sections and shape plot can be regenerated
    # TODO: add save results

def shapePlot(cell,shapeplotsinfo, axsettings,save_flag, figdir = '.', extension = '.png'):
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

def setup_recordTraces(recordTraces,cell,Dt):

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

def setup_recordAPs(h,recordAPs,cell, threshold, preorder = False, colorkeyval=colorkeyval):
    apcounts = []
    aptimevectors = []
    segnames = []
    colorlist = []
    
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
    else:
        raise ValueError('recordAPs: can only be "all" or "all0.5"')
    for seg in seglist:
        segnames.append(str(seg).split('.',1)[-1])
        timevector = h.Vector()
        apc = h.APCount(seg)
        apc.thresh = threshold
        apc.record(timevector)
        apcounts.append(apc)
        aptimevectors.append(timevector)

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
    return tuple(apcounts), tuple(aptimevectors), apinfo

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

    secnr = float(sec.split('[',1)[-1][:-1])
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

if __name__ == '__main__':
    import Functions.setup as stp
    import matplotlib.pyplot as plt

    # create dummy field
    xlims = [-1000,1000]
    ylims = [-1000,1000]
    zlims = [-1000,1000]
    nx= 100
    ny = 100
    nz = 100
    xX,xY,xZ = np.meshgrid(np.linspace(xlims[0],xlims[1],nx),np.linspace(ylims[0],ylims[1],ny), np.linspace(zlims[0],zlims[1],nz),indexing='ij')
    data = np.array((xX.ravel(),xY.ravel(),xZ.ravel())).T
    #myfun = lambda x,y,z: x**2+y**2+(z-2.5)
    myfun = np.vectorize(lambda x,y,z,a: 1 if ((z>=0 and z<=10) and (x**2+y**2)<100**2) else (10/z)**(2/a) if (z>10 and (x**2+y**2)<100**2) else ((10/z)**(2/a)*100**(2/a)/(x**2+y**2)**(1/a)) if z>10 else ((1/(z+10))**(2/a)*100**(2/a)/(x**2+y**2)**(1/a)) if (z>0 and z<=10) else 1e-6 )
    field = np.hstack((data,1000*myfun(data[:,0],data[:,1],data[:,2],1)[:,None]))




    input = stp.simParams({'duration':10, 'test_flag':True,'save_flag': True, 'plot_flag': False, 'stimopt':{'stim_type':['Optogxstim']}})
    input.cellsopt.neurontemplate = Cells.NeuronTemplates[7]

    input.stimopt.Ostimparams.field = eF.prepareDataforInterp(field,'ninterp')
    input.stimopt.Ostimparams.amp = 5000
    input.stimopt.Ostimparams.delay = 100
    input.stimopt.Ostimparams.pulseType = 'pulseTrain'
    input.stimopt.Ostimparams.dur = 600-1e-6
    input.stimopt.Ostimparams.options = {'prf':0.01,'dc':0.5, 'phi': np.pi/2, 'xT': [0,0,100]}
    

    input.stimopt.Estimparams.filepath = 'Inputs\ExtracellularPotentials\Reference - recessed\PotentialDistr-600um-20umMESH_refined_masked_structured.txt'
    input.stimopt.Estimparams.delay = 50
    input.stimopt.Estimparams.dur = 10
    input.stimopt.Estimparams.options['phi'] = np.pi/2
    input.stimopt.Estimparams.options['xT'] = [0,0,100]
    input.cellsopt.extracellular = True

    input.analysesopt.shapeplots.append(dict(cvals_type='es'))



    input.cellsopt.opsin_options.opsinlocations = 'all'
    optogeneticStimulation(input, verbose = True)

    #if input.plot_flag:
     #   plt.show()


