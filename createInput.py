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
    folder = './Inputs/test%s/'%dt_string
    os.makedirs(folder, exist_ok=True)

    filenameopt = folder + f'test.json'
    filenameopt = './Inputs/test.json'

    # create dummy field
    xlims = [-1000,1000]
    ylims = [-1000,1000]
    zlims = [-1000,1000]
    nx= 101
    ny = 101
    nz = 101
    fieldtype = '2D'
    #myfun = lambda x,y,z: x**2+y**2+(z-2.5)
    xX,xY,xZ = np.meshgrid(np.linspace(xlims[0],xlims[1],nx),np.linspace(ylims[0],ylims[1],ny), np.linspace(zlims[0],zlims[1],nz),indexing='ij')
    myfun = np.vectorize(lambda x,y,z,a: 1 if ((z>=0 and z<=10) and (x**2+y**2)<100**2) else (10/z)**(2/a) if (z>10 and (x**2+y**2)<100**2) else ((10/z)**(2/a)*100**(2/a)/(x**2+y**2)**(1/a)) if z>10 else ((1/(z+10))**(2/a)*100**(2/a)/(x**2+y**2)**(1/a)) if (z>0 and z<=10) else 1e-6 )
    # 3D field
    if fieldtype == '3D':
        data = np.array((xX.ravel(),xY.ravel(),xZ.ravel())).T
        field = np.hstack((data/10,1000*myfun(data[:,0],data[:,1],data[:,2],1)[:,None]))
    else :
        idx = int(np.ceil(ny/2)-1)
        xX = xX[:,idx,:]
        xY = xY[:,idx,:]
        xZ= xZ[:,idx,:]
        data = np.array((xX.ravel(),xY.ravel(),xZ.ravel())).T
        field = np.hstack((data/10,1000*myfun(data[:,0],data[:,1],data[:,2],1)[:,None]))
        field = field[:,[0,2,3]]





    input = stp.simParams({'duration':200, 'test_flag':True,'save_flag': True, 'plot_flag': False})

    input.stimopt.stim_type = ['Optogxstim','eVstim']
    input.cellsopt.neurontemplate = Cells.NeuronTemplates[0]
    input.simulationType = ['normal','SD_eVstim','SD_Optogenx','VTA_Optogenx']
    input.cellsopt.opsin_options.opsinlocations = 'apicalnoTuft'
    input.cellsopt.opsin_options.Gmax_total = None #uS
    input.cellsopt.opsin_options.distribution = lambda x: 1000*(np.exp(-np.linalg.norm(np.array(x)-[0,0,0])/200))
    input.v0 = -70

    input.stimopt.Ostimparams.field = eF.prepareDataforInterp(field,'ninterp')
    input.stimopt.Ostimparams.amp = 10
    input.stimopt.Ostimparams.delay = 100
    input.stimopt.Ostimparams.pulseType = 'pulseTrain'
    input.stimopt.Ostimparams.dur = 50
    input.stimopt.Ostimparams.options = {'prf':1/2000,'dc':1/20000, 'phi': np.pi/2, 'xT': [0,0,100]}


    input.stimopt.Estimparams.filepath = 'Inputs\ExtracellularPotentials\Reference - recessed\PotentialDistr-600um-20umMESH_structured.txt'#'Inputs\ExtracellularPotentials\Reference - recessed\PotentialDistr-600um-20umMESH_refined_masked_structured.txt'
    input.stimopt.Estimparams.delay = input.duration+50
    input.stimopt.Estimparams.dur = 10
    input.stimopt.Estimparams.options['phi'] = np.pi/2
    input.stimopt.Estimparams.options['xT'] = [0,0,250]
    input.cellsopt.extracellular = True

    input.analysesopt.shapeplots.append(dict(cvals_type='es'))

    input.analysesopt.SDeVstim.options['simdur']=200
    input.analysesopt.SDeVstim.options['delay']=100
    input.analysesopt.SDeVstim.options['vinit']=-70
    input.analysesopt.SDeVstim.options['n_iters']=2
    input.analysesopt.SDeVstim.durs = np.array([1e0,2e0])

    input.analysesopt.SDOptogenx.options['simdur']=200
    input.analysesopt.SDOptogenx.options['delay']=100
    input.analysesopt.SDOptogenx.options['vinit']=-70
    input.analysesopt.SDOptogenx.options['n_iters']=2
    input.analysesopt.SDOptogenx.options['verbose'] = True
    input.analysesopt.SDOptogenx.startamp = 1000
    input.analysesopt.SDOptogenx.durs = np.logspace(-2,1,2)
    input.analysesopt.SDOptogenx.options['dc_sdc'] = input.analysesopt.SDOptogenx.durs/10
    input.analysesopt.SDOptogenx.nr_pulseOI = 2

    input.analysesopt.VTAOptogenx.options['simdur']=150
    input.analysesopt.VTAOptogenx.options['delay']=50
    input.analysesopt.VTAOptogenx.options['vinit']=-70
    input.analysesopt.VTAOptogenx.options['n_iters']=3
    input.analysesopt.VTAOptogenx.options['verbose'] = True
    input.analysesopt.VTAOptogenx.options['scale_initsearch'] = 4
    input.analysesopt.VTAOptogenx.searchdir = np.array([1,0,0])
    input.analysesopt.VTAOptogenx.startpos = np.array([np.zeros(5)+24,np.zeros(5),np.arange(0,50,10)+10]).T

    input.analysesopt.succesRatioOptions['window'] = 100
    
    with open(filenameopt,'w') as file:
        json.dump(input.todict(False), file, indent = 4, cls=MyEncoder)
