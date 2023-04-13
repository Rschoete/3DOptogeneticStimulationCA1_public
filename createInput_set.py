
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Functions.globalFunctions.ExtracellularField as eF
import Functions.setup as stp
from Functions.globalFunctions.utils import Dict, MyEncoder
from Model import Cells


def _main_first_datasets():
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d")
    folder = './Inputs/SDC%s_Pyr5NNLarge470inVivoGray/' % dt_string
    os.makedirs(folder, exist_ok=True)
    runlist_filenameoptall = folder+'runlistopt'

    inputsoptall = []

    Celltemplates = Cells.NeuronTemplates[:1]
    # opsinlocs_pyrs = #['soma','axon','all','alldend','apic','basal','apicaltrunk','apicaltrunk_ext','apicaltuft','obliques','apicalnotuft']
    # ,'alldend','apic','basal','apicalnotuft']
    opsinlocs_pyrs = ['all', 'soma', 'apic', 'basal']
    opsinlocs_interss = ['all', 'soma', 'alldend']
    Gmaxs = [None]+list(np.logspace(-1, 1, 5))
    distributions = [
        f'distribution = lambda x: {y}' for y in list(np.logspace(1, 3, 5))]
    pds = [0.01, 0.1, 1, 10, 100]
    dcs = [1/100, 1/10, 1/2]

    for cell in Celltemplates:
        if 'pc' in cell.lower():
            opsinlocs = opsinlocs_pyrs
        else:
            opsinlocs = opsinlocs_interss
        for opsinloc in opsinlocs:
            for nrp in [1, 2]:
                dcs_intm = dcs if nrp > 1 else [1]
                iter = -1
                sublist = []
                for Gmax in Gmaxs:
                    if Gmax is None:
                        distrs = distributions
                    else:
                        distrs = [f'distribution = lambda x: {1}']
                    for distr in distrs:
                        for dc in dcs_intm:

                            iter += 1

                            filenameopt = folder + \
                                f'input{cell}_{opsinloc}_np{nrp}_{iter}.json'
                            inputsoptall.append(filenameopt)
                            sublist.append(filenameopt)

                            input = stp.simParams({'test_flag': False, 'save_data_flag': False,
                                                  'save_input_flag': False, 'save_flag': True, 'plot_flag': False})

                            input.resultsFolder = '/SDC_NN470Large_gray_invitro/' + \
                                f'{cell}_{opsinloc}_np{nrp}_{iter}'
                            input.subfolderSuffix = ''

                            input.duration = 100
                            input.v0 = -70

                            input.stimopt.stim_type = ['Optogxstim']
                            input.simulationType = ['SD_Optogenx']

                            input.cellsopt.neurontemplate = cell

                            input.cellsopt.opsin_options.opsinlocations = opsinloc
                            input.cellsopt.opsin_options.Gmax_total = Gmax  # uS
                            input.cellsopt.opsin_options.distribution = distr
                            # allign axo-somato-dendritic axis with z-axis
                            input.cellsopt.init_options.theta = -np.pi/2

                            input.stimopt.Ostimparams.filepath = 'Inputs/LightIntensityProfile/NeuroNexus470Large_gray_invivo_np1e7_res5emin3_cyl_5x10_gf1.txt'
                            input.stimopt.Ostimparams.amp = 1/3*10**5
                            input.stimopt.Ostimparams.delay = 100
                            input.stimopt.Ostimparams.pulseType = 'pulseTrain'
                            input.stimopt.Ostimparams.dur = 10 * 10/dc
                            input.stimopt.Ostimparams.options = {
                                'prf': dc/10, 'dc': dc, 'xT': [0, 0, 0]}

                            input.analysesopt.SDOptogenx.options['simdur'] = 500
                            input.analysesopt.SDOptogenx.options['delay'] = 100
                            input.analysesopt.SDOptogenx.options['vinit'] = -70
                            input.analysesopt.SDOptogenx.options['n_iters'] = 7
                            input.analysesopt.SDOptogenx.options['verbose'] = False
                            input.analysesopt.SDOptogenx.startamp = 1000
                            input.analysesopt.SDOptogenx.durs = pds
                            input.analysesopt.SDOptogenx.options['dc_sdc'] = dc
                            input.analysesopt.SDOptogenx.nr_pulseOI = nrp
                            input.analysesopt.SDOptogenx.record_iOptogenx = 'chr2h134r'

                            input.analysesopt.recordSuccesRatio = False
                            input.analysesopt.sec_plot_flag = False
                            input.analysesopt.save_traces = False

                            with open(filenameopt, 'w') as file:
                                json.dump(input.todict(False), file,
                                          indent=4, cls=MyEncoder)

                with open(folder+'sublist'+cell+opsinloc+str(nrp)+'.csv', 'w') as file:
                    file.write('inputfilename\n')
                    for item in sublist:
                        file.write("%s\n" % item)

    with open(runlist_filenameoptall+'.txt', 'w') as file:
        for item in inputsoptall:
            file.write("%s\n" % item)
    with open(runlist_filenameoptall+'.csv', 'w') as file:
        file.write('inputfilename\n')
        for item in inputsoptall:
            file.write("%s\n" % item)


def _main_const_intensity_single_pulse():
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d")
    folder = './Inputs/SDC%s_constI/' % dt_string
    os.makedirs(folder, exist_ok=True)
    runlist_filenameoptall = folder+'runlistopt'

    inputsoptall = []

    Celltemplates = Cells.NeuronTemplates[:2]+Cells.NeuronTemplates[4:6]
    # opsinlocs_pyrs = #['soma','axon','all','alldend','apic','basal','apicaltrunk','apicaltrunk_ext','apicaltuft','obliques','apicalnotuft']
    # ,'alldend','apic','basal','apicalnotuft']
    opsinlocs_pyrs = ['all', 'soma', 'axon', 'basal', 'alldend', 'apic']
    opsinlocs_interss = ['all', 'soma', 'alldend', 'axon']
    Gmaxs = []  # see in loop below
    Gmaxs_must = list(np.logspace(-1, 1.5, 8))
    pds = list(np.logspace(-1, 3, 9))

    for cell in Celltemplates:
        if 'pc' in cell.lower():
            opsinlocs = opsinlocs_pyrs
        else:
            opsinlocs = opsinlocs_interss
        for opsinloc in opsinlocs:
            if opsinloc in ['soma', 'axon']:
                Gmaxs = list(
                    np.unique(list(np.logspace(-2, 1, 10))+Gmaxs_must))
            else:
                Gmaxs = list(
                    np.unique(list(np.logspace(-1, 2, 10))+Gmaxs_must))
            iter = -1
            sublist = []
            for Gmax in Gmaxs:
                iter += 1

                filenameopt = folder+f'input{cell}_{opsinloc}_{iter}.json'
                inputsoptall.append(filenameopt)
                sublist.append(filenameopt)

                input = stp.simParams({'test_flag': False, 'save_data_flag': True,
                                      'save_input_flag': True, 'save_flag': True, 'plot_flag': False})

                input.resultsFolder = '/SDC_constI/' + \
                    f'{cell}_{opsinloc}_{iter}'
                input.subfolderSuffix = ''

                input.duration = 100
                input.v0 = -70

                input.stimopt.stim_type = ['Optogxstim']
                input.simulationType = ['SD_Optogenx']

                input.cellsopt.neurontemplate = cell

                input.cellsopt.opsin_options.opsinlocations = opsinloc
                input.cellsopt.opsin_options.Gmax_total = Gmax  # uS
                input.cellsopt.opsin_options.distribution = f'distribution = lambda x: {1}'
                # allign axo-somato-dendritic axis with z-axis
                input.cellsopt.init_options.theta = -np.pi/2
                input.cellsopt.init_options.replace_axon = False

                input.stimopt.Ostimparams.filepath = 'Inputs/LightIntensityProfile/constant.txt'
                input.stimopt.Ostimparams.amp = 1/3*10**5
                input.stimopt.Ostimparams.delay = 100
                input.stimopt.Ostimparams.pulseType = 'singleSquarePulse'
                input.stimopt.Ostimparams.dur = 100

                input.analysesopt.SDOptogenx.options['simdur'] = 500
                input.analysesopt.SDOptogenx.options['delay'] = 100
                input.analysesopt.SDOptogenx.options['vinit'] = -70
                input.analysesopt.SDOptogenx.options['n_iters'] = 7
                input.analysesopt.SDOptogenx.options['verbose'] = False
                input.analysesopt.SDOptogenx.startamp = 1000
                input.analysesopt.SDOptogenx.durs = pds
                input.analysesopt.SDOptogenx.nr_pulseOI = 1
                input.analysesopt.SDOptogenx.record_iOptogenx = 'chr2h134r'

                input.analysesopt.recordSuccesRatio = False
                input.analysesopt.sec_plot_flag = False
                input.analysesopt.save_traces = False
                input.analysesopt.save_rasterplot = False
                input.analysesopt.save_SDplot = False
                input.analysesopt.save_shapeplots = False
                input.analysesopt.save_VTAplot = False

                with open(filenameopt, 'w') as file:
                    json.dump(input.todict(False), file,
                              indent=4, cls=MyEncoder)

            with open(folder+'sublist'+cell+opsinloc+'.csv', 'w') as file:
                file.write('inputfilename\n')
                for item in sublist:
                    file.write("%s\n" % item)

    with open(runlist_filenameoptall+'.txt', 'w') as file:
        for item in inputsoptall:
            file.write("%s\n" % item)
    with open(runlist_filenameoptall+'.csv', 'w') as file:
        file.write('inputfilename\n')
        for item in inputsoptall:
            file.write("%s\n" % item)


def _main_const_intensity_single_pulse_differentmorpho_pyrs():
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d")
    folder = './Inputs/SDC%s_constI_diffMorphoPyr/' % dt_string
    os.makedirs(folder, exist_ok=True)
    runlist_filenameoptall = folder+'runlistopt'

    inputsoptall = []

    Celltemplates = Cells.NeuronTemplates[:2]
    morphos = ["mpg141209_A_idA.asc", "mpg141208_B_idA.asc"]
    # opsinlocs_pyrs = #['soma','axon','all','alldend','apic','basal','apicaltrunk','apicaltrunk_ext','apicaltuft','obliques','apicalnotuft']
    # ,'alldend','apic','basal','apicalnotuft']
    opsinlocs_pyrs = ['all', 'soma', 'axon', 'basal', 'alldend', 'apic']
    opsinlocs_interss = ['all', 'soma', 'alldend', 'axon']
    Gmaxs = []  # see in loop below
    Gmaxs_must = list(np.logspace(-1, 1.5, 8))
    pds = list(np.logspace(-1, 3, 9))

    for cell, morpho in zip(Celltemplates, morphos):
        if 'pc' in cell.lower():
            opsinlocs = opsinlocs_pyrs
        else:
            opsinlocs = opsinlocs_interss
        for opsinloc in opsinlocs:
            if opsinloc in ['soma', 'axon']:
                Gmaxs = list(
                    np.unique(list(np.logspace(-2, 1, 10))+Gmaxs_must))
            else:
                Gmaxs = list(
                    np.unique(list(np.logspace(-1, 2, 10))+Gmaxs_must))
            iter = -1
            sublist = []
            for Gmax in Gmaxs:
                iter += 1

                filenameopt = folder+f'input{cell}_{opsinloc}_{iter}.json'
                inputsoptall.append(filenameopt)
                sublist.append(filenameopt)

                input = stp.simParams({'test_flag': False, 'save_data_flag': True,
                                      'save_input_flag': True, 'save_flag': True, 'plot_flag': False})

                input.resultsFolder = '/SDC_constI_diffMorphoPyr/' + \
                    f'{cell}_{opsinloc}_{iter}'
                input.subfolderSuffix = ''

                input.duration = 100
                input.v0 = -70

                input.stimopt.stim_type = ['Optogxstim']
                input.simulationType = ['SD_Optogenx']

                input.cellsopt.neurontemplate = cell

                input.cellsopt.opsin_options.opsinlocations = opsinloc
                input.cellsopt.opsin_options.Gmax_total = Gmax  # uS
                input.cellsopt.opsin_options.distribution = f'distribution = lambda x: {1}'
                # allign axo-somato-dendritic axis with z-axis
                input.cellsopt.init_options.theta = -np.pi/2
                input.cellsopt.init_options.replace_axon = False
                input.cellsopt.init_options.morphology = morpho

                input.stimopt.Ostimparams.filepath = 'Inputs/LightIntensityProfile/constant.txt'
                input.stimopt.Ostimparams.amp = 1/3*10**5
                input.stimopt.Ostimparams.delay = 100
                input.stimopt.Ostimparams.pulseType = 'singleSquarePulse'
                input.stimopt.Ostimparams.dur = 100

                input.analysesopt.SDOptogenx.options['simdur'] = 500
                input.analysesopt.SDOptogenx.options['delay'] = 100
                input.analysesopt.SDOptogenx.options['vinit'] = -70
                input.analysesopt.SDOptogenx.options['n_iters'] = 7
                input.analysesopt.SDOptogenx.options['verbose'] = False
                input.analysesopt.SDOptogenx.startamp = 1000
                input.analysesopt.SDOptogenx.durs = pds
                input.analysesopt.SDOptogenx.nr_pulseOI = 1
                input.analysesopt.SDOptogenx.record_iOptogenx = 'chr2h134r'

                input.analysesopt.recordSuccesRatio = False
                input.analysesopt.sec_plot_flag = False
                input.analysesopt.save_traces = False
                input.analysesopt.save_rasterplot = False
                input.analysesopt.save_SDplot = False
                input.analysesopt.save_shapeplots = False
                input.analysesopt.save_VTAplot = False

                with open(filenameopt, 'w') as file:
                    json.dump(input.todict(False), file,
                              indent=4, cls=MyEncoder)

            with open(folder+'sublist'+cell+opsinloc+'.csv', 'w') as file:
                file.write('inputfilename\n')
                for item in sublist:
                    file.write("%s\n" % item)

    with open(runlist_filenameoptall+'.txt', 'w') as file:
        for item in inputsoptall:
            file.write("%s\n" % item)
    with open(runlist_filenameoptall+'.csv', 'w') as file:
        file.write('inputfilename\n')
        for item in inputsoptall:
            file.write("%s\n" % item)


def _main_fixed_field_different_setups_xposssplit():
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d")
    folder = './Inputs/SDC%s_Ugent470inVivoGray_multiCells_singlePulse_xposssplit/' % dt_string
    os.makedirs(folder, exist_ok=True)
    runlist_filenameoptall = folder+'runlistopt'

    inputsoptall = []
    Celltemplates = Cells.NeuronTemplates[:2]+Cells.NeuronTemplates[4:6]
    length_soma_dend_axis = [751.3, 670.7, 638.7, 553.38]
    ratio_short_long = [2.72, 3.38, 0.91, 1.74]
    # opsinlocs_pyrs = #['soma','axon','all','alldend','apic','basal','apicaltrunk','apicaltrunk_ext','apicaltuft','obliques','apicalnotuft']
    # ,'alldend','apic','basal','apicalnotuft']
    opsinlocs_pyrs = ['all', 'soma', 'basal', 'axon']
    opsinlocs_interss = ['all', 'soma', 'axon']
    Gmaxs = list(np.logspace(-1, 1.5, 8))
    pds = np.logspace(0, 2, 5)
    pitches = [-np.pi/2, 0, np.pi/2]
    ztrans = [-700, -1000, -400]
    xtrans = [0, -700, 0]
    xposs_list = []
    yposs_list = []
    zposs_list = []

    # xlims = [0, 2500]
    # dxmax = 600
    # dxmin = 100
    # xmax = 2500
    # xintm = 20

    # def refinefun1(x): return (dxmax-dxmin)/xmax*abs(x) + \
    #     dxmin if(abs(x) <= xmax) else dxmax
    # xposs_radial = eF.plot_EcF._refinedaxis_xlim(
    #     xlims, refinefun=refinefun1, symmetric=False, plot=False, clip=True)
    xposs_radial = list(np.linspace(0, 2500, 11, endpoint=True))
    xposs_x = list(np.linspace(0, 1100, 11, endpoint=True))
    zposs_radial = list(np.linspace(0, 1100, 11, endpoint=True))
    zposs_x = list(np.linspace(0, 5000, 11, endpoint=True))

    for cell in Celltemplates:
        if 'pc' in cell.lower():
            opsinlocs = opsinlocs_pyrs
        else:
            opsinlocs = opsinlocs_interss
        sublist = []
        xposs_sublist = []
        yposs_sublist = []
        zposs_sublist = []
        for opsinloc in opsinlocs:
            iter = -1
            for pitch, xt, zt in zip(pitches, xtrans, ztrans):
                for Gmax in Gmaxs:
                    iter += 1

                    filenameopt = folder + \
                        f'input{cell}_{opsinloc}_pi{pitch/np.pi:0.1f}_{iter}.json'

                    input = stp.simParams({'test_flag': False, 'save_data_flag': False,
                                           'save_input_flag': False, 'save_flag': False, 'plot_flag': False})

                    input.resultsFolder = '/SDC_Ugent470_gray_invivo_multicell_singlePulse/' + \
                        f'{cell}_{opsinloc}_{iter}'
                    input.subfolderSuffix = ''

                    input.duration = 100
                    input.v0 = -70

                    input.stimopt.stim_type = ['Optogxstim']
                    input.simulationType = ['SD_Optogenx']

                    input.cellsopt.neurontemplate = cell

                    input.cellsopt.opsin_options.opsinlocations = opsinloc
                    input.cellsopt.opsin_options.Gmax_total = Gmax  # uS
                    input.cellsopt.opsin_options.distribution = f'distribution = lambda x: {1}'
                    # allign axo-somato-dendritic axis with z-axis
                    input.cellsopt.init_options.theta = pitch
                    input.cellsopt.init_options.replace_axon = False
                    input.cellsopt.cellTrans_options.move_flag = True
                    input.cellsopt.cellTrans_options.rt = [xt, 0, zt]

                    input.stimopt.Ostimparams.filepath = './Inputs/LightIntensityProfile/Ugent470nIrr_np1e7_res5emin3_gf1_cyl_5x10.txt'
                    input.stimopt.Ostimparams.amp = 1/3*10**5
                    input.stimopt.Ostimparams.delay = 100
                    input.stimopt.Ostimparams.pulseType = 'singleSquarePulse'
                    input.stimopt.Ostimparams.dur = 100
                    input.stimopt.Ostimparams.options = {
                        'prf': 100, 'dc': 1, 'xT': [0, 0, 0]}

                    input.analysesopt.SDOptogenx.options['simdur'] = 500
                    input.analysesopt.SDOptogenx.options['delay'] = 100
                    input.analysesopt.SDOptogenx.options['vinit'] = -70
                    input.analysesopt.SDOptogenx.options['n_iters'] = 7
                    input.analysesopt.SDOptogenx.options['verbose'] = False
                    input.analysesopt.SDOptogenx.startamp = 1000
                    input.analysesopt.SDOptogenx.durs = pds
                    input.analysesopt.SDOptogenx.nr_pulseOI = 1
                    input.analysesopt.SDOptogenx.record_iOptogenx = 'chr2h134r'

                    input.analysesopt.recordSuccesRatio = False
                    input.analysesopt.sec_plot_flag = False
                    input.analysesopt.save_traces = False

                    with open(filenameopt, 'w') as file:
                        json.dump(input.todict(False), file,
                                  indent=4, cls=MyEncoder)

                    if pitch == 0:
                        xposs = xposs_x
                        zposs = zposs_x
                    else:
                        xposs = xposs_radial
                        zposs = zposs_radial
                    yposs = [0]
                    for xp in xposs:
                        inputsoptall.append(filenameopt)
                        sublist.append(filenameopt)
                        xposs_sublist.append([xp])
                        yposs_sublist.append(yposs)
                        zposs_sublist.append(zposs)

        df = pd.DataFrame({'inputfilename': sublist, 'xposs': xposs_sublist,
                           'yposs': yposs_sublist, 'zposs': zposs_sublist})
        df.to_csv(folder+'sublist'+cell+opsinloc+'.csv', index=False)
        xposs_list.extend(xposs_sublist)
        yposs_list.extend(yposs_sublist)
        zposs_list.extend(zposs_sublist)

    with open(runlist_filenameoptall+'.txt', 'w') as file:
        for item in inputsoptall:
            file.write("%s\n" % item)

    df = pd.DataFrame({'inputfilename': inputsoptall,
                      'xposs': xposs_list, 'yposs': yposs_list, 'zposs': zposs_list})
    df.to_csv(runlist_filenameoptall+'.csv', index=False)


def _main_fixed_field_different_setups():
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d")
    folder = './Inputs/SDC%s_Ugent470inVivoGray_multiCells_singlePulse/' % dt_string
    os.makedirs(folder, exist_ok=True)
    runlist_filenameoptall = folder+'runlistopt'

    inputsoptall = []
    Celltemplates = Cells.NeuronTemplates[:2]+Cells.NeuronTemplates[4:6]
    length_soma_dend_axis = [751.3, 670.7, 638.7, 553.38]
    ratio_short_long = [2.72, 3.38, 0.91, 1.74]
    # opsinlocs_pyrs = #['soma','axon','all','alldend','apic','basal','apicaltrunk','apicaltrunk_ext','apicaltuft','obliques','apicalnotuft']
    # ,'alldend','apic','basal','apicalnotuft']
    opsinlocs_pyrs = ['all', 'soma', 'basal', 'axon']
    opsinlocs_interss = ['all', 'soma', 'axon']
    Gmaxs = list(np.logspace(-1, 1.5, 8))
    pds = np.logspace(0, 2, 5)
    pitches = [-np.pi/2, 0, np.pi/2]
    ztrans = [-700, -1000, -400]
    xtrans = [0, -700, 0]
    xposs_list = []
    yposs_list = []
    zposs_list = []

    # xlims = [0, 2500]
    # dxmax = 600
    # dxmin = 100
    # xmax = 2500
    # xintm = 20

    # def refinefun1(x): return (dxmax-dxmin)/xmax*abs(x) + \
    #     dxmin if(abs(x) <= xmax) else dxmax
    # xposs_radial = eF.plot_EcF._refinedaxis_xlim(
    #     xlims, refinefun=refinefun1, symmetric=False, plot=False, clip=True)
    xposs_radial = list(np.linspace(0, 2500, 11, endpoint=True))
    xposs_x = list(np.linspace(0, 1100, 11, endpoint=True))
    zposs_radial = list(np.linspace(0, 1100, 11, endpoint=True))
    zposs_x = list(np.linspace(0, 5000, 11, endpoint=True))

    for cell in Celltemplates:
        if 'pc' in cell.lower():
            opsinlocs = opsinlocs_pyrs
        else:
            opsinlocs = opsinlocs_interss
        for opsinloc in opsinlocs:
            iter = -1
            sublist = []
            xposs_sublist = []
            yposs_sublist = []
            zposs_sublist = []
            for pitch, xt, zt in zip(pitches, xtrans, ztrans):
                for Gmax in Gmaxs:
                    iter += 1

                    filenameopt = folder + \
                        f'input{cell}_{opsinloc}_pi{pitch/np.pi:0.1f}_{iter}.json'
                    inputsoptall.append(filenameopt)
                    sublist.append(filenameopt)

                    input = stp.simParams({'test_flag': False, 'save_data_flag': False,
                                           'save_input_flag': False, 'save_flag': True, 'plot_flag': False})

                    input.resultsFolder = '/SDC_Ugent470_gray_invivo_multicell_singlePulse/' + \
                        f'{cell}_{opsinloc}_{iter}'
                    input.subfolderSuffix = ''

                    input.duration = 100
                    input.v0 = -70

                    input.stimopt.stim_type = ['Optogxstim']
                    input.simulationType = ['SD_Optogenx']

                    input.cellsopt.neurontemplate = cell

                    input.cellsopt.opsin_options.opsinlocations = opsinloc
                    input.cellsopt.opsin_options.Gmax_total = Gmax  # uS
                    input.cellsopt.opsin_options.distribution = f'distribution = lambda x: {1}'
                    # allign axo-somato-dendritic axis with z-axis
                    input.cellsopt.init_options.theta = pitch
                    input.cellsopt.init_options.replace_axon = False
                    input.cellsopt.cellTrans_options.move_flag = True
                    input.cellsopt.cellTrans_options.rt = [xt, 0, zt]

                    input.stimopt.Ostimparams.filepath = './Inputs/LightIntensityProfile/Ugent470nIrr_np1e7_res5emin3_gf1_cyl_5x10.txt'
                    input.stimopt.Ostimparams.amp = 1/3*10**5
                    input.stimopt.Ostimparams.delay = 100
                    input.stimopt.Ostimparams.pulseType = 'singleSquarePulse'
                    input.stimopt.Ostimparams.dur = 100
                    input.stimopt.Ostimparams.options = {
                        'prf': 100, 'dc': 1, 'xT': [0, 0, 0]}

                    input.analysesopt.SDOptogenx.options['simdur'] = 500
                    input.analysesopt.SDOptogenx.options['delay'] = 100
                    input.analysesopt.SDOptogenx.options['vinit'] = -70
                    input.analysesopt.SDOptogenx.options['n_iters'] = 7
                    input.analysesopt.SDOptogenx.options['verbose'] = False
                    input.analysesopt.SDOptogenx.startamp = 1000
                    input.analysesopt.SDOptogenx.durs = pds
                    input.analysesopt.SDOptogenx.nr_pulseOI = 1
                    input.analysesopt.SDOptogenx.record_iOptogenx = 'chr2h134r'

                    input.analysesopt.recordSuccesRatio = False
                    input.analysesopt.sec_plot_flag = False
                    input.analysesopt.save_traces = False

                    if pitch == 0:
                        xposs_sublist.append(xposs_x)
                        zposs_sublist.append(zposs_x)
                    else:
                        xposs_sublist.append(xposs_radial)
                        zposs_sublist.append(zposs_radial)
                    yposs_sublist.append([0])
                    with open(filenameopt, 'w') as file:
                        json.dump(input.todict(False), file,
                                  indent=4, cls=MyEncoder)
            df = pd.DataFrame({'inputfilename': sublist, 'xposs': xposs_sublist,
                              'yposs': yposs_sublist, 'zposs': zposs_sublist})
            df.to_csv(folder+'sublist'+cell+opsinloc+'.csv', index=False)
            xposs_list.extend(xposs_sublist)
            yposs_list.extend(yposs_sublist)
            zposs_list.extend(zposs_sublist)

    with open(runlist_filenameoptall+'.txt', 'w') as file:
        for item in inputsoptall:
            file.write("%s\n" % item)

    df = pd.DataFrame({'inputfilename': inputsoptall,
                      'xposs': xposs_list, 'yposs': yposs_list, 'zposs': zposs_list})
    df.to_csv(runlist_filenameoptall+'.csv', index=False)


def _main_EET_multicells_allparams_xposssplit():
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d")
    folder = './Inputs/SDC%s_Ugent470inVivoGray_multiCells_singlePulse_EET_xposssplit/' % dt_string
    os.makedirs(folder, exist_ok=True)
    runlist_filenameoptall = folder+'runlistopt'

    OAT_samples_EET = pd.read_csv(
        './Inputs/samples_EET_multicell_in_opticalField.csv')
    inputsoptall = []
    pds = np.logspace(0, 2, 5)
    pitches = [-np.pi/2, 0, np.pi/2]
    ztrans = [-700, -1000, -400]
    xtrans = [0, -700, 0]
    xposs_list = []
    yposs_list = []
    zposs_list = []

    xposs_radial = list(np.linspace(0, 2500, 11, endpoint=True))
    xposs_x = list(np.linspace(0, 1100, 11, endpoint=True))
    zposs_radial = list(np.linspace(0, 1100, 11, endpoint=True))
    zposs_x = list(np.linspace(0, 5000, 11, endpoint=True))

    for i in range(len(OAT_samples_EET)):
        sublist = []
        xposs_sublist = []
        yposs_sublist = []
        zposs_sublist = []

        cell = OAT_samples_EET['cell'].iloc[i]
        mua = OAT_samples_EET['mua'].iloc[i]
        mus = OAT_samples_EET['mus'].iloc[i]
        g = OAT_samples_EET['g'].iloc[i]
        Gmax = OAT_samples_EET['Gmax'].iloc[i]
        opsinloc = OAT_samples_EET['loc'].iloc[i]
        roll = OAT_samples_EET['roll'].iloc[i]
        pitch = OAT_samples_EET['pitch'].iloc[i]
        sim_idx = OAT_samples_EET['sim_idx'].iloc[i]
        iter = sim_idx

        filenameopt = folder + \
            f'input_{sim_idx}.json'

        input = stp.simParams({'test_flag': False, 'save_data_flag': False,
                               'save_input_flag': False, 'save_flag': False, 'plot_flag': False})

        input.resultsFolder = '/SDC_Ugent470_gray_invivo_multicell_EET/' + \
            f'EETsimulation_{sim_idx}'
        input.subfolderSuffix = ''

        input.duration = 100
        input.v0 = -70

        input.stimopt.stim_type = ['Optogxstim']
        input.simulationType = ['SD_Optogenx']

        input.cellsopt.neurontemplate = cell

        input.cellsopt.opsin_options.opsinlocations = opsinloc
        input.cellsopt.opsin_options.Gmax_total = Gmax  # uS
        input.cellsopt.opsin_options.distribution = f'distribution = lambda x: {1}'
        # allign axo-somato-dendritic axis with z-axis
        input.cellsopt.init_options.theta = pitch
        input.cellsopt.init_options.psi = roll
        input.cellsopt.init_options.replace_axon = False
        input.cellsopt.cellTrans_options.move_flag = True
        input.cellsopt.cellTrans_options.rt = [xt, 0, zt]

        input.stimopt.Ostimparams.filepath = './Inputs/LightIntensityProfile/Ugent470nIrr_np1e7_res5emin3_gf1_cyl_5x10.txt'
        input.stimopt.Ostimparams.amp = 1/3*10**5
        input.stimopt.Ostimparams.delay = 100
        input.stimopt.Ostimparams.pulseType = 'singleSquarePulse'
        input.stimopt.Ostimparams.dur = 100
        input.stimopt.Ostimparams.options = {
            'prf': 100, 'dc': 1, 'xT': [0, 0, 0]}

        input.analysesopt.SDOptogenx.options['simdur'] = 500
        input.analysesopt.SDOptogenx.options['delay'] = 100
        input.analysesopt.SDOptogenx.options['vinit'] = -70
        input.analysesopt.SDOptogenx.options['n_iters'] = 7
        input.analysesopt.SDOptogenx.options['verbose'] = False
        input.analysesopt.SDOptogenx.startamp = 1000
        input.analysesopt.SDOptogenx.durs = pds
        input.analysesopt.SDOptogenx.nr_pulseOI = 1
        input.analysesopt.SDOptogenx.record_iOptogenx = 'chr2h134r'

        input.analysesopt.recordSuccesRatio = False
        input.analysesopt.sec_plot_flag = False
        input.analysesopt.save_traces = False

        with open(filenameopt, 'w') as file:
            json.dump(input.todict(False), file,
                      indent=4, cls=MyEncoder)

        if pitch == 0:
            xposs = xposs_x
            zposs = zposs_x
        else:
            xposs = xposs_radial
            zposs = zposs_radial
        yposs = [0]
        for xp in xposs:
            inputsoptall.append(filenameopt)
            sublist.append(filenameopt)
            xposs_sublist.append([xp])
            yposs_sublist.append(yposs)
            zposs_sublist.append(zposs)

        df = pd.DataFrame({'inputfilename': sublist, 'xposs': xposs_sublist,
                           'yposs': yposs_sublist, 'zposs': zposs_sublist})
        df.to_csv(folder+'sublist'+cell+opsinloc+'.csv', index=False)
        xposs_list.extend(xposs_sublist)
        yposs_list.extend(yposs_sublist)
        zposs_list.extend(zposs_sublist)

    with open(runlist_filenameoptall+'.txt', 'w') as file:
        for item in inputsoptall:
            file.write("%s\n" % item)

    df = pd.DataFrame({'inputfilename': inputsoptall,
                      'xposs': xposs_list, 'yposs': yposs_list, 'zposs': zposs_list})
    df.to_csv(runlist_filenameoptall+'.csv', index=False)


def _main_EET_multicells_allparams_pitchcelltypesplit_xposssplit():

    celltemplates_pyrs = Cells.NeuronTemplates[:2]
    opsinlocs_pyrs = ['all', 'soma', 'basal', 'axon']
    celltemplates_ints = Cells.NeuronTemplates[4:6]
    opsinlocs_ints = ['all', 'soma', 'axon']

    now = datetime.now()
    dt_string = now.strftime("%Y%m%d")
    folder = './Inputs/SDC%s_Ugent470inVivoGray_multiCells_singlePulse_celltypepitchsplit_EET_xposssplit/' % dt_string
    os.makedirs(folder, exist_ok=True)
    runlist_filenameoptall = folder+'runlistopt'

    OAT_samples_EET = pd.read_csv(
        './Inputs/samples_EET_multicell_in_opticalField_pitchcelltypesplit.csv')
    inputsoptall = []
    pds = np.logspace(0, 2, 5)
    pitches = [-np.pi/2, 0, np.pi/2]
    ztrans = [-700, -1000, -400]
    xtrans = [0, -700, 0]
    xposs_list = []
    yposs_list = []
    zposs_list = []

    xposs_radial = list(np.linspace(0, 2500, 11, endpoint=True))
    xposs_x = list(np.linspace(0, 1100, 11, endpoint=True))
    zposs_radial = list(np.linspace(0, 1100, 11, endpoint=True))
    zposs_x = list(np.linspace(0, 5000, 11, endpoint=True))
    for celltype in ['pyr', 'int']:
        OAT_samples_EET = pd.read_csv(
            f'./Inputs/samples_EET_multicell_in_opticalField_pitchcelltypesplit{celltype}_v2.csv')
        sublist = []
        xposs_sublist = []
        yposs_sublist = []
        zposs_sublist = []
        if celltype == 'pyr':
            celltemplates = celltemplates_pyrs
            opsinlocs = opsinlocs_pyrs
        elif celltype == 'int':
            celltemplates = celltemplates_ints
            opsinlocs = opsinlocs_ints

        for pitch, xt, zt in zip(pitches, xtrans, ztrans):
            for i in range(len(OAT_samples_EET)):

                cell = OAT_samples_EET['cell'].iloc[i].astype(int)
                mua = OAT_samples_EET['mua'].iloc[i]
                mus = OAT_samples_EET['mus'].iloc[i]
                g = OAT_samples_EET['g'].iloc[i]
                redus = mus*(1-g)
                Gmax = OAT_samples_EET['Gmax'].iloc[i]
                opsinloc = OAT_samples_EET['loc'].iloc[i].astype(int)
                if celltype == 'pyr':
                    if opsinloc > 3:
                        raise ValueError()
                        opsinloc = int(np.floor(opsinloc/3))
                elif celltype == 'int':
                    if opsinloc > 2:
                        raise ValueError()
                        opsinloc = int(np.floor(opsinloc/4))
                roll = OAT_samples_EET['roll'].iloc[i]-np.pi
                sim_idx = OAT_samples_EET['sim_idx'].iloc[i]
                # map sim_idx to idx optical field
                ofield_idx = 3*np.floor(sim_idx/7).astype(int)
                if sim_idx % 7 == 1:
                    ofield_idx += 1
                if sim_idx % 7 == 2:
                    ofield_idx += 2
                opticfield_filepath = f'./Inputs/LightIntensityProfile/Ugent470grayinvivo_EET/ugent470_gray_invivo_mua{mua:0.4f}_redmus{redus:0.4f}_{ofield_idx}.txt'
                opsinloc = opsinlocs[opsinloc]
                cell = celltemplates[int(cell)]

                filenameopt = folder + \
                    f'input{celltype}_pitch{pitch/np.pi:0.1f}_{sim_idx}.json'

                input = stp.simParams({'test_flag': False, 'save_data_flag': False,
                                       'save_input_flag': False, 'save_flag': False, 'plot_flag': False})

                input.resultsFolder = '/SDC_Ugent470_gray_invivo_multicell_pitchcelltypesplit_EET/' + \
                    f'EETsimulation{celltype}_pitch{pitch/np.pi:0.1f}_{sim_idx}'
                input.subfolderSuffix = ''

                input.duration = 100
                input.v0 = -70

                input.stimopt.stim_type = ['Optogxstim']
                input.simulationType = ['SD_Optogenx']

                input.cellsopt.neurontemplate = cell

                input.cellsopt.opsin_options.opsinlocations = opsinloc
                input.cellsopt.opsin_options.Gmax_total = Gmax  # uS
                input.cellsopt.opsin_options.distribution = f'distribution = lambda x: {1}'
                # allign axo-somato-dendritic axis with z-axis
                input.cellsopt.init_options.theta = pitch
                input.cellsopt.init_options.psi = roll
                input.cellsopt.init_options.replace_axon = False
                input.cellsopt.cellTrans_options.move_flag = True
                input.cellsopt.cellTrans_options.rt = [xt, 0, zt]

                input.stimopt.Ostimparams.filepath = opticfield_filepath
                input.stimopt.Ostimparams.amp = 1/3*10**5
                input.stimopt.Ostimparams.delay = 100
                input.stimopt.Ostimparams.pulseType = 'singleSquarePulse'
                input.stimopt.Ostimparams.dur = 100
                input.stimopt.Ostimparams.options = {
                    'prf': 100, 'dc': 1, 'xT': [0, 0, 0]}

                input.analysesopt.SDOptogenx.options['simdur'] = 500
                input.analysesopt.SDOptogenx.options['delay'] = 100
                input.analysesopt.SDOptogenx.options['vinit'] = -70
                input.analysesopt.SDOptogenx.options['n_iters'] = 7
                input.analysesopt.SDOptogenx.options['verbose'] = False
                input.analysesopt.SDOptogenx.startamp = 1000
                input.analysesopt.SDOptogenx.durs = pds
                input.analysesopt.SDOptogenx.nr_pulseOI = 1
                input.analysesopt.SDOptogenx.record_iOptogenx = 'chr2h134r'

                input.analysesopt.recordSuccesRatio = False
                input.analysesopt.sec_plot_flag = False
                input.analysesopt.save_traces = False

                with open(filenameopt, 'w') as file:
                    json.dump(input.todict(False), file,
                              indent=4, cls=MyEncoder)

                if pitch == 0:
                    xposs = xposs_x
                    zposs = zposs_x
                else:
                    xposs = xposs_radial
                    zposs = zposs_radial
                yposs = [0]
                for xp in xposs:
                    inputsoptall.append(filenameopt)
                    sublist.append(filenameopt)
                    xposs_sublist.append([xp])
                    yposs_sublist.append(yposs)
                    zposs_sublist.append(zposs)

        df = pd.DataFrame({'inputfilename': sublist, 'xposs': xposs_sublist,
                           'yposs': yposs_sublist, 'zposs': zposs_sublist})
        df.to_csv(folder+'sublist'+celltype+'.csv', index=False)
        xposs_list.extend(xposs_sublist)
        yposs_list.extend(yposs_sublist)
        zposs_list.extend(zposs_sublist)

    with open(runlist_filenameoptall+'.txt', 'w') as file:
        for item in inputsoptall:
            file.write("%s\n" % item)

    df = pd.DataFrame({'inputfilename': inputsoptall,
                      'xposs': xposs_list, 'yposs': yposs_list, 'zposs': zposs_list})
    df.to_csv(runlist_filenameoptall+'.csv', index=False)


if __name__ == '__main__':
    _main_const_intensity_single_pulse_differentmorpho_pyrs()
    # _main_const_intensity_single_pulse()
    print('finish')
