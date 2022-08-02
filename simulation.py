import os
import sys
import time
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
import Functions.globalFunctions.ExtracellularField as eF
import Functions.support as sprt
import Functions.globalFunctions.featExtract as featE

def optogeneticStimulation(input, verbose = False):
    print("\n\nDate and time =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    if not isinstance(input.simulationType,list):
        input.simulationType = [input.simulationType]

    if not input.plot_flag:
        # With this backend plots are not shown but can be saved
        mpluse("AGG")
    else:
        mpluse("TKAGG")

    # important for reproducibility, seed based on ceratain date
    seed = input.seed
    rng = np.random.seed(int(seed))
    random.seed(int(seed))

    # start timer
    timerstart = time.time()

    # Cell setup
    # ----------------------------------------------------------
    print('\nCell setup')
    ##Load cell
    if input.cellsopt.neurontemplate in Cells.NeuronTemplates:
        print(f'Loading cell: {input.cellsopt.neurontemplate}')
        cell = getattr(Cells,input.cellsopt.neurontemplate)(**input.cellsopt.init_options)
        print(f'\t* celltype: {cell.celltype}\n\t* morphology: {cell.morphology}')
    else:
        raise ValueError(f"input.neuronInput = {input.cellsopt.neurontemplate} is invalid. Possible templates are  {Cells.NeuronTemplates}")


    ## Plugin opsin
    oopt = input.cellsopt.opsin_options
    if oopt.opsinlocations is not None:
        print(f'\t* opsin: {oopt.opsinmech}')
        if isinstance(oopt.opsinlocations,str):
            print(f'\t* opsinlocations: {oopt.opsinlocations}')
            # if opsin location is a string convert to list of sections (method of neurontemplate class)
            oopt.opsinlocations = sprt.convert_strtoseclist(cell,oopt.opsinlocations)

        cell.insertOptogenetics(seclist = oopt.opsinlocations, opsinmech = oopt.opsinmech, set_pointer_xtra =  oopt.set_pointer_xtra)

        if oopt.distribution_source_hdistance is not None:
            oopt.distribution_source_hdistance = sprt.convert_strtoseg(oopt.distribution_source_hdistance)
        seglist, values = cell.distribute_mechvalue(oopt.distribution,method=oopt.distribution_method,source_hdistance = oopt.distribution_source_hdistance)
        cell.updateMechValue(seglist,values,'gchr2bar_'+oopt.opsinmech)


    ## Plugin extracellular
    print(f"\t* extracellular: {input.cellsopt.extracellular}")
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
    print('\nLoadingFields')
    ## Load fields
    # Potential field
    estim_amp = []; estim_time = []; totales = 0
    if 'eVstim' in input.stimopt.stim_type:
        print('\t* extracellular electrical stimulation')
        params = input.stimopt.Estimparams
        if params['field'] is None:
            field = np.genfromtxt(params['filepath'],comments='%')
            params['field'] = eF.prepareDataforInterp(field, params['method_prepareData'])
            del field
        t = np.arange(0,input.duration*1.1, input.dt/10)
        estim_time,estim_amp,totales = eF.setXstim(cell.allsec, t, params['delay'], params['dur'], params['amp'], params['field'],params['structured'],params['pulseType'],stimtype='electrical',netpyne=False, **params['options'])


    # Optical field
    ostim_amp = []; ostim_time = []; totalos = 0
    if 'Optogxstim' in input.stimopt.stim_type:
        print('\t* Optogenetics stimulation')
        params = input.stimopt.Ostimparams
        if params['field'] is None:
            field = np.genfromtxt(params['filepath'],comments='%')
            params['field'] = eF.prepareDataforInterp(field, params['method_prepareData'])
            del field
        t = np.arange(0,input.duration*1.1, input.dt/10)
        ostim_time,ostim_amp,totalos = eF.setXstim(cell.allsec, t, params['delay'], params['dur'], params['amp'], params['field'],params['structured'],params['pulseType'],stimtype = 'optical',netpyne=False, **params['options'])
    print('')

    # Perform simulations
    # ----------------------------------------------------------
    if any([x in input.simulationType for x in ['SD_eVstim','SD_Optogenx']]):
        print('Calculating Strength Duration Curves')
    amps_SDeVstim = None; amps_SDoptogenx = None
    if 'SD_eVstim' in input.simulationType:
        print('\t* SD_eVstim')
        SDcopt = input.analysesopt.SDeVstim
        params = input.stimopt.Estimparams
        SDcopt.cellrecloc = sprt.convert_strtoseg(cell,SDcopt.cellrecloc)
        amps_SDeVstim = featE.SD_curve_xstim(h,cell,SDcopt.durs,params['field'],SDcopt.startamp,SDcopt.cellrecloc,SDcopt.stimtype,stimpointer=SDcopt.stimpointer,nr_pulseOI=SDcopt.nr_pulseOI,estimoptions=params['options'], **SDcopt.options)
        print('\t  ',[f"{dur:0.2e} -> {amp:0.2e}" if amp is not None else f"{dur:0.2e} -> None" for dur,amp in zip(SDcopt.durs,amps_SDeVstim)],'\n')

    if  'SD_Optogenx' in input.simulationType:
        print('\t* SD_Optogenx')
        SDcopt = input.analysesopt.SDOptogenx
        params = input.stimopt.Ostimparams
        SDcopt.cellrecloc = sprt.convert_strtoseg(cell,SDcopt.cellrecloc)
        amps_SDoptogenx = featE.SD_curve_xstim(h,cell,SDcopt.durs,params['field'],SDcopt.startamp,SDcopt.cellrecloc,SDcopt.stimtype,stimpointer=SDcopt.stimpointer,nr_pulseOI=SDcopt.nr_pulseOI,estimoptions=params['options'], **SDcopt.options)
        print('\t  ',[f"{dur:0.2e} -> {amp:0.2e}" if amp is not None else f"{dur:0.2e} -> None" for dur,amp in zip(SDcopt.durs,amps_SDoptogenx)],'\n')


    t=None; vsoma=None; traces = None; apcounts = None; aptimevectors=None; apinfo=None
    if 'normal' in input.simulationType:

        if estim_amp is not None and len(estim_amp)>0:
            estim_amp = h.Vector(estim_amp)
            estim_time = h.Vector(estim_time)
            estim_amp.play(h._ref_estim_xtra, estim_time, True) #True -> interpolate

        if ostim_amp is not None and len(ostim_amp)>0:
            # need to do at this level (does not work when in do not know why)
            ostim_amp = h.Vector(ostim_amp)
            ostim_time = h.Vector(ostim_time)
            ostim_amp.play(h._ref_ostim_xtra, ostim_time, True) #True -> interpolate
        print('Normal Simulation')
        # Analyses setup
        # ----------------------------------------------------------
        ## setup recording traces
        print(f'Setup Analyses\n\t* sampling frequency: {input.samplingFrequency} Hz\n')
        aopt = input.analysesopt
        Dt = 1/input.samplingFrequency*1e3
        t, vsoma, traces = sprt.setup_recordTraces(h,aopt.recordTraces,cell,Dt)
        if aopt.recordTotalOptogeneticCurrent:
            traces['iOptogenx']= {'traces':[], 'names':[]}
            opsinmech = input.cellsopt.opsin_options.opsinmech
            for sec in cell.allsec:
                for seg in sec:
                    if hasattr(seg,opsinmech):
                        traces['iOptogenx']['traces'].append(h.Vector().record(getattr(seg,f'_ref_i_{opsinmech}'),Dt))
                        traces['iOptogenx']['names'].append(seg)
        apcounts, aptimevectors, apinfo, idx_sR = sprt.setup_recordAPs(h,aopt.recordAPs,cell, threshold=aopt.apthresh,succesRatio_seg= aopt.succesRatioOptions['succesRatio_seg'], preorder= aopt.preordersecforplot)


        # Simulate
        # ----------------------------------------------------------
        print('Simulating')
        print(f'\t* dt: {input.dt} ms\n\t* duration: {input.duration} ms\n...')
        timer_startsim = time.time()
        h.dt = input.dt
        h.celsius = input.celsius
        h.finitialize(input.v0)
        h.continuerun(input.duration)
        timer_stopsim = time.time()
        print(f"simulation finished in {timer_stopsim-timer_startsim:0.2f} s\n")

    # Create folders:
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M")
    results_dir = "./Results%s/Results_%s_%s_%s"%(input.resultsFolder, input.subfolderSuffix,input.cellsopt.neurontemplate,dt_string)
    fig_dir = results_dir+"/Figures"
    if input.save_flag:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

    # Analyse
    # ----------------------------------------------------------
    print('Analyses')
    # Create
    iOptogenx, succes_ratio = sprt.AnalysesWrapper(h,input,cell,t,vsoma,traces,ostim_time,ostim_amp,estim_time,estim_amp,aptimevectors,apinfo,idx_sR,amps_SDeVstim,amps_SDoptogenx,fig_dir)

    # stop timer
    timerstop = time.time()
    print(f'\nTotal time: {timerstop-timerstart:.2f} s\n')

    # Saving Data
    # ----------------------------------------------------------
    print('Saving Data')
    inputdata,data = sprt.SaveResults(input,cell,t,vsoma,traces,apcounts,aptimevectors,apinfo,totales,totalos, iOptogenx, succes_ratio,amps_SDeVstim,amps_SDoptogenx,timerstop-timerstart,seed,results_dir)
    return inputdata, data
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




    input = stp.simParams({'duration':4500, 'test_flag':True,'save_flag': True, 'plot_flag': True})

    input.stimopt.stim_type = ['Optogxstim']
    input.cellsopt.neurontemplate = Cells.NeuronTemplates[0]
    input.simulationType = ['normal']
    input.cellsopt.opsin_options.opsinlocations = 'apicalnoTuft'
    input.v0 = -70

    input.stimopt.Ostimparams.field = eF.prepareDataforInterp(field,'ninterp')
    input.stimopt.Ostimparams.amp = 600.2109
    input.stimopt.Ostimparams.delay = 100
    input.stimopt.Ostimparams.pulseType = 'pulseTrain'
    input.stimopt.Ostimparams.dur = 4000-1e-6
    input.stimopt.Ostimparams.options = {'prf':1/2000,'dc':1/20000, 'phi': np.pi/2, 'xT': [0,0,100]}


    input.stimopt.Estimparams.filepath = 'Inputs\ExtracellularPotentials\Reference - recessed\PotentialDistr-600um-20umMESH_structured.txt'#'Inputs\ExtracellularPotentials\Reference - recessed\PotentialDistr-600um-20umMESH_refined_masked_structured.txt'
    input.stimopt.Estimparams.delay = 100
    input.stimopt.Estimparams.dur = 10
    input.stimopt.Estimparams.options['phi'] = np.pi/2
    input.stimopt.Estimparams.options['xT'] = [0,0,250]
    input.cellsopt.extracellular = True

    input.analysesopt.shapeplots.append(dict(cvals_type='es'))

    input.analysesopt.SDeVstim.options['simdur']=200
    input.analysesopt.SDeVstim.options['delay']=100
    input.analysesopt.SDeVstim.options['vinit']=-70
    input.analysesopt.SDeVstim.options['n_iters']=1
    input.analysesopt.SDeVstim.durs = np.array([1e0,2e0])

    input.analysesopt.SDOptogenx.options['simdur']=1000
    input.analysesopt.SDOptogenx.options['delay']=100
    input.analysesopt.SDOptogenx.options['vinit']=-70
    input.analysesopt.SDOptogenx.options['n_iters']=7
    input.analysesopt.SDOptogenx.options['verbose'] = True
    input.analysesopt.SDOptogenx.durs = np.logspace(-3,3,7)
    input.analysesopt.SDOptogenx.options['dc_sdc'] = input.analysesopt.SDOptogenx.durs/2000
    input.analysesopt.SDOptogenx.nr_pulseOI = 2

    input.analysesopt.succesRatioOptions['window'] = 100

    optogeneticStimulation(input, verbose = True)

    if input.plot_flag:
        plt.show()


