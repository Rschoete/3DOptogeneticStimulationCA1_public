from importlib.metadata import distribution


if __name__ == '__main__':
    import Functions.setup as stp
    import matplotlib.pyplot as plt
    import numpy as np
    from Model import Cells
    import Functions.globalFunctions.ExtracellularField as eF
    from Functions.globalFunctions.utils import MyEncoder,Dict
    import json
    import os
    from datetime import datetime

    now = datetime.now()
    dt_string = now.strftime("%Y%m%d")
    folder = './Inputs/SDC%s/'%dt_string
    os.makedirs(folder, exist_ok=True)
    runlist_filenameoptall = folder+'runlistopt'

    inputsoptall = []

    Celltemplates = Cells.NeuronTemplates[:4]
    #opsinlocs_pyrs = #['soma','axon','all','alldend','apic','basal','apicaltrunk','apicaltrunk_ext','apicaltuft','obliques','apicalnotuft']
    opsinlocs_pyrs = ['all','soma','apic']#,'alldend','apic','basal','apicalnotuft']
    opsinlocs_interss = ['all','soma']#,'alldend']
    Gmaxs = [None]+list(np.logspace(-1,1,5))
    distributions = [f'distribution = lambda x: {y}' for y in list(np.logspace(1,3,5))]
    rots = list(2/12*np.pi*np.arange(4))
    xposs = list(np.arange(0,2000,200))
    zposs = list(np.arange(0,5000,500))
    pds = [0.01,0.1,1,10,100]
    dcs = [1/100,1/10,1/2]

    for nrp  in [1,2]:
        for cell in Celltemplates:
            sublist = []
            if 'pyr' in cell.lower():
                opsinlocs = opsinlocs_pyrs
            else:
                opsinlocs = opsinlocs_interss
            for opsinloc in opsinlocs:
                iter = -1
                for Gmax in Gmaxs:
                    if Gmax is None:
                        distrs = distributions
                    else:
                        distrs = [f'distribution = lambda x: {1}']
                    for distr in distrs:
                        for dc in dcs:

                            iter+=1

                            filenameopt = folder+f'input{cell}_{opsinloc}_np{nrp}_{iter}.json'
                            inputsoptall.append(filenameopt)
                            sublist.append(filenameopt)

                            input = stp.simParams({'test_flag':False,'save_flag': True, 'plot_flag': False})

                            input.resultsFolder = '/SDC'
                            input.subfolderSuffix = f'{cell}_{opsinloc}_np{nrp}_{iter}'

                            input.stimopt.stim_type = ['Optogxstim']
                            input.simulationType = ['normal','SD_Optogenx']

                            input.cellsopt.neurontemplate = cell

                            input.cellsopt.opsin_options.opsinlocations = opsinloc
                            input.cellsopt.opsin_options.Gmax_total = Gmax #uS
                            input.cellsopt.opsin_options.distribution = distr
                            input.cellsopt.init_options.theta = 0

                            input.duration = 100+10 * 10/dc+100
                            input.v0 = -70


                            input.stimopt.Ostimparams.filepath = 'Inputs/LightIntensityProfile/Ugent470nIrr_np1e7_res5emin3_gf1_cyl_5x10.txt'
                            input.stimopt.Ostimparams.amp = 1/3*10**5
                            input.stimopt.Ostimparams.delay = 100
                            input.stimopt.Ostimparams.pulseType = 'pulseTrain'
                            input.stimopt.Ostimparams.dur = 10 * 10/dc
                            input.stimopt.Ostimparams.options = {'prf':dc/10,'dc':dc, 'psi': np.pi/2, 'xT': [0,0,0]}

                            input.analysesopt.SDOptogenx.options['simdur']=500
                            input.analysesopt.SDOptogenx.options['delay']=100
                            input.analysesopt.SDOptogenx.options['vinit']=-70
                            input.analysesopt.SDOptogenx.options['n_iters']=7
                            input.analysesopt.SDOptogenx.options['verbose'] = False
                            input.analysesopt.SDOptogenx.startamp = 1000
                            input.analysesopt.SDOptogenx.durs = pds
                            input.analysesopt.SDOptogenx.options['dc_sdc'] = dc
                            input.analysesopt.SDOptogenx.nr_pulseOI = nrp


                            input.analysesopt.succesRatioOptions['window'] = 100
                            input.analysesopt.sec_plot_flag = False
                            input.analysesopt.recordTraces = {}
                            input.analysesopt.save_traces = False

                            with open(filenameopt,'w') as file:
                                json.dump(input.todict(False), file, indent = 4, cls=MyEncoder)
            
            with open(folder+'sublist'+cell+str(nrp)+'.csv','w') as file:
                file.write('inputfilename\n')
                for  item in sublist:
                    file.write("%s\n" % item)


    with open(runlist_filenameoptall+'.txt','w') as file:
        for  item in inputsoptall:
            file.write("%s\n" % item)
    with open(runlist_filenameoptall+'.csv','w') as file:
        file.write('inputfilename\n')
        for  item in inputsoptall:
            file.write("%s\n" % item)
