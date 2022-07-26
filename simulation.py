import os
import sys
import time
import random
from matplotlib.pyplot import tight_layout
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

def optogeneticStimulation(input, verbose = False):

    if not input.plot_flag:
        # With this backend plots are not shown but can be saved
        mpluse("AGG")
    else:
        mpluse("TKAGG")

    # important for reproducibility, seed based on ceratain date
    seed = input.seed
    rng = np.random.seed(int(seed))
    random.seed(int(seed))

    #Load cell
    if input.cellsopt.neurontemplate in Cells.NeuronTemplates:
        cell = getattr(Cells,input.cellsopt.neurontemplate)(**input.cellsopt.init_options)
        if verbose:
            print(f'{input.cellsopt.neurontemplate} loaded')
    else:
        raise ValueError(f"input.neuronInput = {input.cellsopt.neurontemplate} is invalid. Possible templates are  {Cells.NeuronTemplates}")

    # Plugin opsin
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

    # Plugin extracellular
    if input.cellsopt.extracellular:
        cell.insertExtracellular()

    # Cell transformation
    # Rotate cell
    ctopt = input.cellsopt.cellTrans_options
    if ctopt.rotate_flag:
        cell.rotate_Cell(ctopt.phi,ctopt.theta)
    # Translate cell
    if ctopt.move_flag:
        cell.move_Cell(ctopt.rt)
    del ctopt

    # assign light stimulus
    cell.updateXtraCoors()

    # check if all pointers set
    cell.check_pointers(input.autosetPointer)

    #Load fields
    # [ ]: test if change to params reflect to input.stimopt.Estimparams as well
    estim_amp = []; estim_time = []
    if 'eVstim' in input.stimopt.stim_type:
        params = input.stimopt.Estimparams
        if params['field'] is None:
            field = np.genfromtxt(params['filepath'],comments='%')
            params['field'] = eF.prepareDataforInterp(field, params['method_prepareData'])
            del field
        t = np.arange(0,input.duration*1.1, input.dt/10)
        estim_time,estim_amp,totales = eF.setEstim(cell.allsec, t, params['delay'], params['dur'], params['amp'], params['field'],params['structured'],params['pulseType'],netpyne=False, stimtype = 'electrical', **params['options'])
        if estim_amp is not None and len(estim_amp)>0:
            # need to do at this level (does not work when in setEstim do not know why)
            estim_amp = h.Vector(estim_amp)
            estim_time = h.Vector(estim_time)
            estim_amp.play(h._ref_estim_xtra, estim_time, True) #True -> interpolate
    
    # [ ]: test if change to params reflect to input.stimopt.Ostimparams as well
    # [ ]: stimtype options works?
    ostim_amp = []; ostim_time = []
    if 'Optogxstim' in input.stimopt.stim_type:
        params = input.stimopt.Ostimparams
        if params['field'] is None:
            field = np.genfromtxt(params['filepath'],comments='%')
            params['field'] = eF.prepareDataforInterp(field, params['method_prepareData'])
            del field
        t = np.arange(0,input.duration*1.1, input.dt/10)
        ostim_time,ostim_amp,totalos = eF.setEstim(cell.allsec, t, params['delay'], params['dur'], params['amp'], params['field'],params['structured'],params['pulseType'],netpyne=False, stimtype = 'optical', **params['options'])
        if ostim_amp is not None and len(ostim_amp)>0:
            # need to do at this level (does not work when in setEstim do not know why)
            ostim_amp = h.Vector(ostim_amp)
            ostim_time = h.Vector(ostim_time)
            ostim_amp.play(h._ref_ostim_xtra, ostim_time, True) #True -> interpolate

    # setup recording
    # TODO: add recording options/refine below

    Dt = 1/input.samplingFrequency*1e3
    t, vsoma, traces = setup_recordTraces(input.analysesopt.recordTraces,cell,Dt)
    

    # do simulation
    h.dt = input.dt
    h.celsius = input.celsius
    h.finitialize(input.v0)
    h.continuerun(input.duration)

    
    # TODO: imporve analyses
    # [ ]: plots of membrane potentials, currents
    # [ ]: shape plots of fields, gbars

    # make some plots to test implementation
    if input.analysesopt.sec_plot_flag:
        cell.sec_plot()
    if input.analysesopt.print_secpos:
        cell.gather_secpos(print_flag = True)
    if len(input.analysesopt.shapeplots)>0:
        for pi in input.analysesopt.shapeplots:
            fig = plt.figure(figsize = (16,10))
            ax = plt.subplot(111,projection='3d')
            mphv2.shapeplot(h,ax,**pi)
            ax.set_title(cell)
            ax.set_zlim([-300,300])
            ax.set_xlim([-300,300])
            ax.set_ylim([-200,400])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.view_init(elev=90, azim=-90)

    plot_traces(t,vsoma,traces,ostim_time,ostim_amp,estim_time,estim_amp)
    
    if input.plot_flag:
        plt.show(block=False)

    print('finished')

    # TODO: add save results, save figures


def plot_traces(t,vsoma,traces,ostim_time,ostim_amp,estim_time,estim_amp):
    t = np.array(t)
    vsoma = np.array(vsoma)
    # plot traces
    fig,ax = plt.subplots(1,1,tight_layout=True,figsize=(9,6))
    ax.plot(t,vsoma)

    # fill locations of pulses
    for stimtime,stimamp,clr in zip([ostim_time,estim_time],[ostim_amp,estim_amp],['tab:blue','tab:yellow']):
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

    # plot all recorded traces
    for k,v in traces.items():

        nrsubplots = int(np.ceil(len(v['traces'])/4))
        nrfigs = int(np.ceil(nrsubplots/3))

        sp_count = 0
        for ifig in range(nrfigs):
            fig = plt.figure(tight_layout=True)
            if sp_count+3<nrsubplots:
                axs = fig.subplots(3,1,sharex=True)
            else:
                axs = fig.subplots(nrsubplots-sp_count,1,sharex=True)
                if not isinstance(axs,np.ndarray):
                    axs = [axs]

            for i,ax in enumerate(axs):
                t1 = ifig*12+i*4
                t2 = ifig*12+(i+1)*4
                for trace, name in zip(v['traces'][t1:t2],v['names'][t1:t2]):
                    ax.plot(t,trace,label=name.split('.')[-1])
                ax.legend()
            fig.suptitle(k)


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




    input = stp.simParams({'plot_flag': True, 'stimopt':{'stim_type':'Optogxstim'}})
    input.stimopt.Ostimparams.field = eF.prepareDataforInterp(field,'ninterp')
    input.stimopt.Ostimparams.options['phi'] = np.pi/2
    input.stimopt.Ostimparams.options['xT'] = [0,0,100]
    input.stimopt.Ostimparams.amp = 5000

    input.cellsopt.opsin_options.opsinlocations = 'apicalnotuft'
    optogeneticStimulation(input, verbose = True)

    if input.plot_flag:
        plt.show()


