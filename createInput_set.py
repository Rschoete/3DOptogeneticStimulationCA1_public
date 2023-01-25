
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

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
    pds = [0.1, 1, 10, 100, 1000]

    for cell in Celltemplates:
        if 'pc' in cell.lower():
            opsinlocs = opsinlocs_pyrs
        else:
            opsinlocs = opsinlocs_interss
        for opsinloc in opsinlocs:
            if opsinloc in ['soma', 'axon']:
                Gmaxs = list(np.logspace(-2, 1, 10))
            else:
                Gmaxs = list(np.logspace(-1, 2, 10))
            iter = -1
            sublist = []
            for Gmax in Gmaxs:
                distr = f'distribution = lambda x: {1}'
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
                input.cellsopt.opsin_options.distribution = distr
                # allign axo-somato-dendritic axis with z-axis
                input.cellsopt.init_options.theta = -np.pi/2

                input.stimopt.Ostimparams.filepath = 'Inputs/LightIntensityProfile/constant.txt'
                input.stimopt.Ostimparams.amp = 1/3*10**5
                input.stimopt.Ostimparams.delay = 100
                input.stimopt.Ostimparams.pulseType = 'pulseTrain'
                input.stimopt.Ostimparams.dur = 100

                input.analysesopt.SDOptogenx.options['simdur'] = 500
                input.analysesopt.SDOptogenx.options['delay'] = 100
                input.analysesopt.SDOptogenx.options['vinit'] = -70
                input.analysesopt.SDOptogenx.options['n_iters'] = 7
                input.analysesopt.SDOptogenx.options['verbose'] = False
                input.analysesopt.SDOptogenx.startamp = 1000
                input.analysesopt.SDOptogenx.durs = pds
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


if __name__ == '__main__':
    _main_const_intensity_single_pulse()
    print('finish')
