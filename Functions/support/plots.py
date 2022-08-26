import matplotlib.pyplot as plt
from ..globalFunctions import morphology_v2 as mphv2
import numpy as np
from .utils import convert_strtoseg, convert_strtoseclist, convert_strtosec
colorkeyval = {'soma':'tab:red', 'axon':'tomato','apical trunk':'tab:blue','apical trunk ext':'royalblue', 'apical tuft': 'tab:green','apical obliques': 'tab:cyan', 'basal dendrites': 'tab:olive', 'unclassified':[0,0,0]}

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
