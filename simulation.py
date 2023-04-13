import os
import random
import sys
import time

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

from matplotlib import use as mpluse

import Functions.globalFunctions.ExtracellularField as eF
import Functions.globalFunctions.featExtract as featE
import Functions.support as sprt
from Model import Cells


def fieldStimulation(input, cell=None, verbose=False, **kwargs):
    print("\n\nDate and time =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    if not isinstance(input.simulationType, list):
        input.simulationType = [input.simulationType]

    if not input.plot_flag:
        # With this backend plots are not shown but can be saved
        mpluse("AGG")
    else:
        try:
            mpluse("TKAGG")
        except Exception as e:
            print(e)

    # important for reproducibility, seed based on ceratain date
    seed = input.seed
    rng = np.random.seed(int(seed))
    random.seed(int(seed))

    # start timer
    timerstart = time.time()

    # Cell setup
    # ----------------------------------------------------------
    print('\nCell setup')
    # Load cell
    if input.cellsopt.neurontemplate in Cells.NeuronTemplates and cell is None:
        print(f'Loading cell: {input.cellsopt.neurontemplate}')
        cell = getattr(Cells, input.cellsopt.neurontemplate)(
            **input.cellsopt.init_options)
        print(
            f'\t* celltype: {cell.celltype}\n\t* morphology: {cell.morphology}')
    elif cell is not None:
        cell.rotate_Cell(inverse=True, init_rotation=True)
        for key in ['movesomatoorigin', 'ID', 'ty', 'col', 'phi', 'theta', 'psi']:
            setattr(cell, key, input.cellsopt.init_options[key])
        cell.moveSomaToOrigin()
        cell.rotate_Cell(init_rotation=True)
    else:
        raise ValueError(
            f"input.neuronInput = {input.cellsopt.neurontemplate} is invalid. Possible templates are  {Cells.NeuronTemplates}")

    # set temperature
    if input.celsius is None:
        print(f'\t* h.celsius = cell.celsius = {cell.celsius}')
        h.celsius = cell.celsius
        input.celsius = cell.celsius
    else:
        print(f'\t* h.celsius = input.celsius = {input.celsius}')
        h.celsius = input.celsius

    # Plugin opsin
    oopt = input.cellsopt.opsin_options
    if oopt.opsinlocations is not None:
        print(f'\t* opsin: {oopt.opsinmech}')
        if isinstance(oopt.opsinlocations, str):
            print(f'\t* opsinlocations: {oopt.opsinlocations}')
            # if opsin location is a string convert to list of sections (method of neurontemplate class)
            oopt.opsinlocations = sprt.convert_strtoseclist(
                cell, oopt.opsinlocations)

        cell.insertOptogenetics(seclist=oopt.opsinlocations,
                                opsinmech=oopt.opsinmech, set_pointer_xtra=oopt.set_pointer_xtra)

        if oopt.distribution_source_hdistance is not None:
            oopt.distribution_source_hdistance = sprt.convert_strtoseg(
                oopt.distribution_source_hdistance)
        seglist, values = cell.distribute_mechvalue(
            oopt.distribution, seclist=oopt.opsinlocations, method=oopt.distribution_method, source_hdistance=oopt.distribution_source_hdistance)
        G_total, seglist, values = cell.calc_Gmax_mechvalue(
            'gchr2bar_'+oopt.opsinmech, values=values, seglist=seglist)
        if oopt.Gmax_total is not None:
            values = [val*oopt.Gmax_total/G_total for val in values]
        else:
            oopt.Gmax_total = G_total
        cell.updateMechValue(seglist, values, 'gchr2bar_'+oopt.opsinmech)

    # Plugin extracellular
    print(f"\t* extracellular: {input.cellsopt.extracellular}")
    if input.cellsopt.extracellular:
        cell.insertExtracellular()

    # Cell transformation
    # Rotate cell
    ctopt = input.cellsopt.cellTrans_options
    if ctopt.rotate_flag:
        cell.rotate_Cell(ctopt.phi, ctopt.theta, ctopt.psi)
    # Translate cell
    if ctopt.move_flag:
        cell.move_Cell(ctopt.rt)
    del ctopt

    cell.updateXtraCoors()  # important to update Xtra values after transformation. values stored in xtra mech are used to calculated received Ve and I

    # check if all pointers set
    # autoset missing pointers to track a non change variable that equals 0 at init -> reason why sections do not receive light when no opsin present (in shape plots)
    cell.check_pointers(input.autosetPointer)

    # Stimulation setup
    # ----------------------------------------------------------
    print('\nLoadingFields')
    # Load fields
    # Potential field
    estim_amp = []
    estim_time = []
    totales = 0
    edts = []
    esimtimes = []
    if 'eVstim' in input.stimopt.stim_type:
        print('\t* extracellular electrical stimulation')
        params = input.stimopt.Estimparams
        if 'eVfield' in kwargs.keys() and kwargs['eVfield'] is not None:
            params['field'] = kwargs['eVfield']
        if params['field'] is None:
            field = np.genfromtxt(params['filepath'], comments='%')
            params['field'] = eF.prepareDataforInterp(
                field, params['method_prepareData'])
            del field
        t = np.arange(0, input.duration*1.1, input.dt/10)
        estim_time, estim_amp, totales = eF.setXstim(cell.allsec, t, params['delay'], params['dur'], params['amp'], params[
                                                     'field'], params['structured'], params['pulseType'], stimtype='electrical', netpyne=False, **params['options'])
        edts, esimtimes = input.stimopt.get_dt_simintm(
            'Estimparams', input.dt, fs=20)

    # Optical field
    ostim_amp = []
    ostim_time = []
    totalos = 0
    odts = []
    osimtimes = []
    if 'Optogxstim' in input.stimopt.stim_type:
        print('\t* Optogenetics stimulation')
        params = input.stimopt.Ostimparams
        if 'Optogxfield' in kwargs.keys() and kwargs['Optogxfield'] is not None:
            params['field'] = kwargs['Optogxfield']
        if params['field'] is None:
            field = np.genfromtxt(params['filepath'], comments='%')
            params['field'] = eF.prepareDataforInterp(
                field, params['method_prepareData'])
            del field
        t = np.arange(0, input.duration*1.1, input.dt/10)
        ostim_time, ostim_amp, totalos = eF.setXstim(cell.allsec, t, params['delay'], params['dur'], params['amp'], params[
                                                     'field'], params['structured'], params['pulseType'], stimtype='optical', netpyne=False, **params['options'])
        odts, osimtimes = input.stimopt.get_dt_simintm(
            'Ostimparams', input.dt, fs=20)
    print('')

    # Perform simulations
    # ----------------------------------------------------------
    if any([x in input.simulationType for x in ['SD_eVstim', 'SD_Optogenx']]):
        print('Calculating Strength Duration Curves')
    amps_SDeVstim = None
    amps_SDoptogenx = None
    if 'SD_eVstim' in input.simulationType:
        print('\t* SD_eVstim')
        SDcopt = input.analysesopt.SDeVstim
        params = input.stimopt.Estimparams
        SDcopt.cellrecloc = sprt.convert_strtoseg(cell, SDcopt.cellrecloc)
        amps_SDeVstim = featE.SD_curve_xstim(h, cell, SDcopt.durs, params['field'], SDcopt.startamp, SDcopt.cellrecloc, SDcopt.stimtype,
                                             stimpointer=SDcopt.stimpointer, nr_pulseOI=SDcopt.nr_pulseOI, estimoptions=params['options'].copy(), return_spikeCountperPulse=SDcopt.return_spikeCountperPulse, **SDcopt.options)
        print('\t  ', [f"{dur:0.2e} -> {amp:0.2e}" if amp is not None else f"{dur:0.2e} -> None" for dur,
              amp in zip(SDcopt.durs, amps_SDeVstim[0])], '\n')

    if 'SD_Optogenx' in input.simulationType:
        print('\t* SD_Optogenx')
        SDcopt = input.analysesopt.SDOptogenx
        params = input.stimopt.Ostimparams
        SDcopt.cellrecloc = sprt.convert_strtoseg(cell, SDcopt.cellrecloc)
        amps_SDoptogenx = featE.SD_curve_xstim(h, cell, SDcopt.durs, params['field'], SDcopt.startamp, SDcopt.cellrecloc, SDcopt.stimtype,
                                               stimpointer=SDcopt.stimpointer, nr_pulseOI=SDcopt.nr_pulseOI, estimoptions=params['options'].copy(), record_iOptogenx=SDcopt.record_iOptogenx, return_spikeCountperPulse=SDcopt.return_spikeCountperPulse, **SDcopt.options)
        print('\t  ', [f"{dur:0.2e} -> {amp:0.2e}" if amp is not None else f"{dur:0.2e} -> None" for dur,
              amp in zip(SDcopt.durs, amps_SDoptogenx[0])], '\n')

    if any([x in input.simulationType for x in ['VTA_eVstim', 'VTA_Optogenx']]):
        print('Calculating VTA')
    pos_VTAeVstim = None
    pos_VTAOptogenx = None
    if 'VTA_eVstim' in input.simulationType:
        print('\t* VTA_eVstim')
        VTAopt = input.analysesopt.VTAeVstim
        params = input.stimopt.Estimparams
        VTAopt.cellrecloc = sprt.convert_strtoseg(cell, VTAopt.cellrecloc)
        pos_VTAeVstim = featE.VTA_xstim(h, cell, VTAopt.startpos, VTAopt.searchdir, params['field'], params['amp'], params['dur'], VTAopt.cellrecloc,
                                        VTAopt.stimtype, stimpointer=VTAopt.stimpointer, nr_pulseOI=VTAopt.nr_pulseOI, estimoptions=params['options'].copy(), **VTAopt.options)
        print(f'\t amp={params["amp"]},dur = {params["dur"]}')
        print('\t  ', [f"{spos} -> {pos}" if pos is not None else f"{spos} -> None" for spos,
              pos in zip(VTAopt.startpos, pos_VTAOptogenx)], '\n')
    if 'VTA_Optogenx' in input.simulationType:
        print('\t* VTA_Optogenx')
        VTAopt = input.analysesopt.VTAOptogenx
        params = input.stimopt.Ostimparams
        VTAopt.cellrecloc = sprt.convert_strtoseg(cell, VTAopt.cellrecloc)
        pos_VTAOptogenx = featE.VTA_xstim(h, cell, VTAopt.startpos, VTAopt.searchdir, params['field'], params['amp'], params['dur'], VTAopt.cellrecloc,
                                          VTAopt.stimtype, stimpointer=VTAopt.stimpointer, nr_pulseOI=VTAopt.nr_pulseOI, estimoptions=params['options'].copy(), **VTAopt.options)
        print(f'\t amp={params["amp"]},dur = {params["dur"]}')
        print('\t  ', [f"{spos} -> {pos}" if pos is not None else f"{spos} -> None" for spos,
              pos in zip(VTAopt.startpos, pos_VTAOptogenx)], '\n')

    t = None
    vsoma = None
    traces = None
    apcounts = None
    aptimevectors = None
    apinfo = None
    idx_sR = None
    if 'normal' in input.simulationType:

        if estim_amp is not None and len(estim_amp) > 0:
            estim_amp = h.Vector(estim_amp)
            estim_time = h.Vector(estim_time)
            estim_amp.play(h._ref_estim_xtra, estim_time,
                           True)  # True -> interpolate

        if ostim_amp is not None and len(ostim_amp) > 0:
            # need to do at this level (does not work when in do not know why)
            ostim_amp = h.Vector(ostim_amp)
            ostim_time = h.Vector(ostim_time)
            ostim_amp.play(h._ref_ostim_xtra, ostim_time,
                           True)  # True -> interpolate
        print('Normal Simulation')
        # Analyses setup
        # ----------------------------------------------------------
        # setup recording traces
        print(
            f'Setup Analyses\n\t* sampling frequency: {input.samplingFrequency} Hz\n')
        aopt = input.analysesopt
        Dt = 1/input.samplingFrequency*1e3
        t, vsoma, traces = sprt.setup_recordTraces(
            h, aopt.recordTraces, cell, Dt)
        if aopt.recordTotalOptogeneticCurrent:
            traces['iOptogenx'] = {'traces': [], 'names': []}
            opsinmech = input.cellsopt.opsin_options.opsinmech
            for sec in cell.allsec:
                for seg in sec:
                    if hasattr(seg, opsinmech):
                        traces['iOptogenx']['traces'].append(
                            h.Vector().record(getattr(seg, f'_ref_i_{opsinmech}'), Dt))
                        traces['iOptogenx']['names'].append(seg)
        apcounts, aptimevectors, apinfo, idx_sR = sprt.setup_recordAPs(
            h, aopt.recordAPs, cell, threshold=aopt.apthresh, succesRatio_seg=aopt.succesRatioOptions['succesRatio_seg'], preorder=aopt.preordersecforplot)

        # Simulate
        # ----------------------------------------------------------
        print('Simulating')
        if input.dt_adapttopd:
            print(
                f'\t* Piecewise simulation total simtime: {input.duration} ms')
            dts, simdurs = sprt.get_dts_simdurs(
                input.duration, input.dt, edts, esimtimes, odts, osimtimes)
            timer_startsim = time.time()
            h.finitialize(input.v0)
            for i, (dt, simdur) in enumerate(zip(dts, simdurs)):
                print(
                    f'\t {i+1}/{len(dts)}, dt: {dt} ms  -  duration: {simdur} ms')
                h.dt = dt
                h.continuerun(simdur)
            timer_stopsim = time.time()
            print(
                f"simulation finished in {timer_stopsim-timer_startsim:0.2f} s\n")
            input.dt = dts
            input.duration = simdurs
        else:
            print(
                f'\t* dt: {input.dt} ms\n\t* duration: {input.duration} ms\n...')
            timer_startsim = time.time()
            h.dt = input.dt
            h.finitialize(input.v0)
            h.continuerun(input.duration)
            timer_stopsim = time.time()
            print(
                f"simulation finished in {timer_stopsim-timer_startsim:0.2f} s\n")

    # Create folders:
    results_dir, fig_dir = sprt.save.createFolder(input)
    if input.save_flag:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

    # Analyse
    # ----------------------------------------------------------
    print('Analyses')
    # Create
    iOptogenx, succes_ratio, VTAOptogenx, VTAeVstim = sprt.AnalysesWrapper(
        h, input, cell, t, vsoma, traces, ostim_time, ostim_amp, estim_time, estim_amp, aptimevectors, apinfo, idx_sR, amps_SDeVstim, amps_SDoptogenx, pos_VTAeVstim, pos_VTAOptogenx, fig_dir)

    # stop timer
    timerstop = time.time()
    print(f'\nTotal time: {timerstop-timerstart:.2f} s\n')

    # Saving Data
    # ----------------------------------------------------------
    print('Saving Data')
    # store fields in new variable before overwritten in SaveResults -> can be passed onto new simulation in **kwargs
    eVfield = input.stimopt.Estimparams['field']
    Optogxfield = input.stimopt.Ostimparams['field']
    inputdata, data = sprt.SaveResults(input, cell, t, vsoma, traces, apcounts, aptimevectors, apinfo, totales, totalos,
                                       iOptogenx, succes_ratio, amps_SDeVstim, amps_SDoptogenx, VTAeVstim, VTAOptogenx, timerstop-timerstart, seed, results_dir)
    return inputdata, data, cell, Optogxfield, eVfield


def gridFieldStimulation(input, xposs=None, yposs=None, zposs=None, overWriteSave=None):
    import Functions.setup as stp
    cell = None
    Optogxfield = None
    firstrun = True
    inputs_all = {}
    data_all = {}
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M")
    timerstart = time.time()
    suffix = ''

    if xposs is None:
        xposs = [0]
    else:
        xposs = xposs if isinstance(xposs, list) else [xposs]
        suffix += f'_x{xposs[0]}'
        if len(xposs) > 1:
            suffix += f'-{xposs[-1]}'

    if yposs is None:
        yposs = list(np.arange(0, 2000, 200))
    else:
        yposs = yposs if isinstance(yposs, list) else [yposs]
        suffix += f'_y{yposs[0]}'
        if len(yposs) > 1:
            suffix += f'-{yposs[-1]}'

    if zposs is None:
        zposs = list(np.arange(0, 5000, 500))
    else:
        zposs = zposs if isinstance(zposs, list) else [zposs]
        suffix += f'_z{zposs[0]}'
        if len(zposs) > 1:
            suffix += f'-{zposs[-1]}'

    for xpos in xposs:
        ypos_succes = np.zeros(len(yposs))
        for iy, ypos in enumerate(yposs):
            zpos_succes = np.zeros(len(zposs))
            for iz, zpos in enumerate(zposs):

                key = f"x{xpos:0.2f}-y{ypos:0.2f}-z{zpos:0.2f}"

                myinput = stp.simParams(input)
                myinput.resultsFolder = myinput.resultsFolder+dt_string
                myinput.subfolderSuffix = key
                myinput.stimopt.Ostimparams.options['xT'] = [
                    float(xpos), float(ypos), float(zpos)]
                if overWriteSave is not None:
                    myinput.save_flag = bool(overWriteSave)

                results_dir, _ = sprt.save.createFolder(myinput)
                myinput, data, cell, Optogxfield, _ = fieldStimulation(
                    myinput, cell=cell, Optogxfield=Optogxfield)

                # Store Info
                if firstrun:
                    firstrun = False
                    if 'recordTraces' in myinput['settings']['analysesopt'].keys():
                        del myinput['settings']['analysesopt']['recordTraces']
                    inputs_all['info'] = myinput

                inputs_all[key] = {'xT': myinput['settings']
                                   ['stimopt']['Ostimparams']['options']['xT']}
                for k in ['phi', 'theta', 'psi']:
                    inputs_all[key][k] = myinput['settings']['cellsopt']['init_options'][k]

                if 'traces' in data.keys():
                    del data['traces']
                if 'eVstim' in data.keys():
                    del data['eVstim']
                data_all[key] = data

                # break loop when no value in SDcurve found
                # next value will result in even lower intensities (we shift neuron away from source)
                if not all([x is None for x in data['SDcurve']['Optogenx']]):
                    zpos_succes[iz] = 1
                if all([x is None for x in data['SDcurve']['Optogenx']]) and iz > 1 and sum(zpos_succes[iz-1:iz+1]) == 0:
                    break
            if sum(zpos_succes) > 0:
                ypos_succes[iy] = 1
            if all([x is None for x in data['SDcurve']['Optogenx']]) and iy > 2 and sum(ypos_succes[iy-1:iy+1]) == 0:
                break

    timerstop = time.time()
    print(f'\nTotal time gridFieldSimulation: {timerstop-timerstart:.2f} s\n')
    print('Saving Results')
    from Functions.support.save import save_data_wCorrectSaveTest
    results_dir = results_dir.rsplit('/', 1)[0]
    os.makedirs(results_dir, exist_ok=True)
    resultsname = results_dir+f'/data{suffix}.json'
    inputsname = results_dir+f'/input{suffix}.json'
    save_data_wCorrectSaveTest(
        resultsname, data_all, test_flag=False, indent=4, signif=myinput['settings']['signif'])
    save_data_wCorrectSaveTest(
        inputsname, inputs_all, test_flag=False, indent=4, signif=myinput['settings']['signif'])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    import Functions.setup as stp
    now = datetime.now()
    cell = None
    Optogxfield = None

    cellname = Cells.NeuronTemplates[0]
    opsinloc = 'all'
    idx = [0, 1, 2, 3, 5, 7, 9, 11, 13, 14]
    GmaxsIrrpd10 = [[0.2276, 0.4642, 0.5179, 1., 1.1788, 2.1544, 2.6827, 4.6416, 6.1054, 10., 13.895, 21.5443, 31.6228, 46.4159, 100.],
                    [4.0631867e+00, 4.2123000e-01, 3.2767360e-01, 9.7869800e-02, 7.6211500e-02, 3.0885000e-02, 2.2383000e-02, 1.0271300e-02, 6.9475000e-03, 3.5438000e-03, 2.2676000e-03, 1.2831000e-03, 7.7960000e-04, 4.7540000e-04, 1.8280000e-04]]
    pd = 10

    iter = -1
    for idx in idx[:]:
        iter += 1

        input = stp.simParams({'test_flag': False, 'save_data_flag': True,
                               'save_input_flag': True, 'save_flag': True, 'plot_flag': False})

        input.resultsFolder = '/Temp/' + \
            f'{pd}{cellname}_{opsinloc}_{iter}'
        input.subfolderSuffix = ''

        input.duration = 200+pd
        input.v0 = -70

        input.stimopt.stim_type = ['Optogxstim']
        input.simulationType = ['normal']

        input.cellsopt.neurontemplate = cellname

        input.cellsopt.opsin_options.opsinlocations = opsinloc
        input.cellsopt.opsin_options.Gmax_total = GmaxsIrrpd10[0][idx]  # uS
        input.cellsopt.opsin_options.distribution = f'distribution = lambda x: {1}'
        # allign axo-somato-dendritic axis with z-axis
        input.cellsopt.init_options.theta = -np.pi/2
        input.cellsopt.init_options.replace_axon = False
        input.cellsopt.init_options.morphology = "mpg141209_A_idA.asc"

        input.stimopt.Ostimparams.filepath = 'Inputs/LightIntensityProfile/constant.txt'
        input.stimopt.Ostimparams.amp = GmaxsIrrpd10[1][idx]*1000
        input.stimopt.Ostimparams.delay = 100
        input.stimopt.Ostimparams.pulseType = 'singleSquarePulse'
        input.stimopt.Ostimparams.dur = pd

        input.analysesopt.recordSuccesRatio = True
        input.analysesopt.sec_plot_flag = False
        input.analysesopt.save_traces = True
        input.analysesopt.save_rasterplot = True
        input.analysesopt.save_SDplot = False
        input.analysesopt.save_shapeplots = True
        input.analysesopt.save_VTAplot = False

        input, data, cell, Optogxfield, _ = fieldStimulation(
            input, cell=cell, Optogxfield=Optogxfield)
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt

#     import Functions.setup as stp

#     # create dummy field
#     xlims = [-1000, 1000]
#     ylims = [-1000, 1000]
#     zlims = [-1000, 1000]
#     nx = 101
#     ny = 101
#     nz = 101
#     fieldtype = '2D'
#     #myfun = lambda x,y,z: x**2+y**2+(z-2.5)
#     xX, xY, xZ = np.meshgrid(np.linspace(xlims[0], xlims[1], nx), np.linspace(
#         ylims[0], ylims[1], ny), np.linspace(zlims[0], zlims[1], nz), indexing='ij')
#     myfun = np.vectorize(lambda x, y, z, a: 1 if ((z >= 0 and z <= 10) and (x**2+y**2) < 100**2) else (10/z)**(2/a) if (z > 10 and (x**2+y**2) < 100**2)
#                          else ((10/z)**(2/a)*100**(2/a)/(x**2+y**2)**(1/a)) if z > 10 else ((1/(z+10))**(2/a)*100**(2/a)/(x**2+y**2)**(1/a)) if (z > 0 and z <= 10) else 1e-6)
#     # 3D field
#     if fieldtype == '3D':
#         data = np.array((xX.ravel(), xY.ravel(), xZ.ravel())).T
#         field = np.hstack(
#             (data/10, 1000*myfun(data[:, 0], data[:, 1], data[:, 2], 1)[:, None]))
#     else:
#         idx = int(np.ceil(ny/2)-1)
#         xX = xX[:, idx, :]
#         xY = xY[:, idx, :]
#         xZ = xZ[:, idx, :]
#         data = np.array((xX.ravel(), xY.ravel(), xZ.ravel())).T
#         field = np.hstack(
#             (data/10, 1000*myfun(data[:, 0], data[:, 1], data[:, 2], 1)[:, None]))

#     # constant field of 1
#     xX, xY, xZ = np.meshgrid(np.linspace(xlims[0], xlims[1], 2), np.linspace(
#         ylims[0], ylims[1], 2), np.linspace(zlims[0], zlims[1], 2), indexing='ij')
#     idx = int(np.ceil(2/2)-1)
#     xX = xX[:, idx, :]
#     xY = xY[:, idx, :]
#     xZ = xZ[:, idx, :]
#     data = np.array((xX.ravel(), xY.ravel(), xZ.ravel())).T
#     field = np.hstack((data*10, np.ones((len(data), 1))))
#     field = field[:, [0, 2, 3]]
#     field = eF.prepareDataforInterp(field, 'ninterp')

#     input = stp.simParams(
#         {'duration': 200, 'test_flag': True, 'save_flag': True, 'plot_flag': True})

#     input.stimopt.stim_type = ['Optogxstim']
#     input.cellsopt.neurontemplate = Cells.NeuronTemplates[0]
#     input.simulationType = ['normal', 'SD_Optogenx']
#     input.cellsopt.opsin_options.opsinlocations = 'apicalnoTuft'
#     input.cellsopt.opsin_options.Gmax_total = None  # uS
#     input.cellsopt.opsin_options.distribution = lambda x: 1000 * \
#         (np.exp(-np.linalg.norm(np.array(x)-[0, 0, 0])/200))
#     input.v0 = -70

#     #input.stimopt.Ostimparams.filepath = "Inputs/LightIntensityProfile/Ugent470_gray_invitro_np1e7_res5emin3_cyl_5x10_gf1.txt"
#     input.stimopt.Ostimparams.filepath = 'Inputs/LightIntensityProfile/constant.txt'
#     input.stimopt.Ostimparams.amp = 10
#     input.stimopt.Ostimparams.delay = 100
#     input.stimopt.Ostimparams.pulseType = 'singleSquarePulse'
#     input.stimopt.Ostimparams.dur = 50
#     input.stimopt.Ostimparams.options = {
#         'prf': 1/2000, 'dc': 1/20000, 'theta': -np.pi/2, 'xT': [0, 0, 100]}

#     input.analysesopt.SDOptogenx.record_iOptogenx = 'chr2h134r'
#     input.analysesopt.SDOptogenx.options['n_iters'] = 2
#     input.analysesopt.SDOptogenx.durs = np.logspace(-2, 1, 2)
#     input.analysesopt.SDOptogenx.startamp = 1000
#     input.analysesopt.SDOptogenx.nr_pulseOI = 2
#     '''
#     input.stimopt.Estimparams.filepath = 'Inputs\ExtracellularPotentials\Reference - recessed\PotentialDistr-600um-20umMESH_structured.txt'#'Inputs\ExtracellularPotentials\Reference - recessed\PotentialDistr-600um-20umMESH_refined_masked_structured.txt'
#     input.stimopt.Estimparams.delay = input.duration+50
#     input.stimopt.Estimparams.dur = 10
#     input.stimopt.Estimparams.options['phi'] = np.pi/2
#     input.stimopt.Estimparams.options['xT'] = [0,0,250]
#     input.cellsopt.extracellular = True

#     input.analysesopt.shapeplots.append(dict(cvals_type='es'))

#     input.analysesopt.SDeVstim.options['simdur']=200
#     input.analysesopt.SDeVstim.options['delay']=100
#     input.analysesopt.SDeVstim.options['vinit']=-70
#     input.analysesopt.SDeVstim.options['n_iters']=2
#     input.analysesopt.SDeVstim.durs = np.array([1e0,2e0])

#     input.analysesopt.SDOptogenx.options['simdur']=200
#     input.analysesopt.SDOptogenx.options['delay']=100
#     input.analysesopt.SDOptogenx.options['vinit']=-70
#     input.analysesopt.SDOptogenx.options['n_iters']=2
#     input.analysesopt.SDOptogenx.options['verbose'] = True
#     input.analysesopt.SDOptogenx.startamp = 1000
#     input.analysesopt.SDOptogenx.durs = np.logspace(-2,1,2)
#     input.analysesopt.SDOptogenx.options['dc_sdc'] = input.analysesopt.SDOptogenx.durs/10
#     input.analysesopt.SDOptogenx.nr_pulseOI = 2

#     input.analysesopt.VTAOptogenx.options['simdur']=150
#     input.analysesopt.VTAOptogenx.options['delay']=50
#     input.analysesopt.VTAOptogenx.options['vinit']=-70
#     input.analysesopt.VTAOptogenx.options['n_iters']=3
#     input.analysesopt.VTAOptogenx.options['verbose'] = True
#     input.analysesopt.VTAOptogenx.options['scale_initsearch'] = 4
#     input.analysesopt.VTAOptogenx.searchdir = np.array([1,0,0])
#     input.analysesopt.VTAOptogenx.startpos = np.array([np.zeros(5)+24,np.zeros(5),[1,5,10,20,30]]).T
#     '''

#     input.analysesopt.succesRatioOptions['window'] = 100

#     fieldStimulation(input, verbose=True)

#     if input.plot_flag:
#         plt.show()
