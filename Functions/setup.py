import inspect
from datetime import datetime

import neuron
import numpy as np

from .globalFunctions.utils import Dict, replaceDictODict


class simParams(object):
    def __init__(self, simParamsDict=None):
        # Flags
        self.test_flag = False
        self.plot_flag = True
        self.save_flag = True
        self.save_data_flag = True
        self.save_input_flag = True
        self.autosetPointer = True
        self.reduce = True

        # 'normal', 'SD_Optogenx', 'SD_eVstim', 'VTA_Optogenx'
        self.simulationType = ['normal']

        self.seed = datetime(2021, 12, 1, 13, 0).timestamp()

        self.duration = 1000
        self.samplingFrequency = 10000  # Hz
        self.dt = 0.025
        self.dt_adapttopd = True
        self.celsius = None  # if none then use cell.celsius
        self.defaultThreshold = -10
        self.v0 = -65

        self.resultsFolder = '/Temp'
        self.subfolderSuffix = ''
        self.signif = 4
        self.resultsmemlim = 50*1024**2

        self.cellsopt = cellsOptions()
        self.stimopt = stimOptions()
        self.analysesopt = analysesOptions()

        if simParamsDict:
            # overwrite defaults
            netParamsComponents = ['cellsopt', 'stimopt', 'analysesopt']
            for k, v in simParamsDict.items():
                if k in netParamsComponents:
                    recursiveDictUpdate(getattr(self, k), v)
                else:
                    if hasattr(self, k):
                        if isinstance(v, dict):
                            setattr(self, k, Dict(v))
                        else:
                            setattr(self, k, v)
                    else:
                        raise AttributeError(k)

    def todict(self, reduce=None, inplace=False):
        # [ ] ISSUE: #1 inplace=False does not same to work anymore
        if not inplace:
            out = replaceDictODict(self.__dict__).copy()
            out = replaceNeuronSectionsandFunTostr(out).copy()
        else:
            out = replaceDictODict(self.__dict__)
            out = replaceNeuronSectionsandFunTostr(out)
        if reduce is None:
            reduce = self.reduce
        if reduce:
            msg = 'removedToReduceMemory'
            # add here variables that need to be substituted by msg
            for x in ['Ostimparams', 'Estimparams']:
                out['stimopt'][x]['field'] = msg
        return out


class cellsOptions(Dict):
    """
    Class to hold options form Model/Cells

    """

    def __init__(self):
        super().__init__()
        # for other options see cells.Neurontemplates
        self.neurontemplate = 'CA1_PC_cAC_sig5'
        # include extracellular mechinism for extracellular electrical stimulation
        self.extracellular = False

        # initialization options
        # options of the neurontemplate class
        self.init_options = Dict()
        self.init_options.replace_axon = True
        self.init_options.morphologylocation = './Model/morphologies'
        self.init_options.ID = 0
        self.init_options.ty = 0
        self.init_options.col = 0
        self.init_options.phi = 0
        self.init_options.theta = 0
        self.init_options.psi = 0
        self.init_options.movesomatoorigin = True

        # cell transformation options
        # !Note!: currently in code first rotation then movement (with init_options movesomatoorigin true soma is at origin)
        self.cellTrans_options = Dict()
        self.cellTrans_options.move_flag = False
        self.cellTrans_options.rt = [0, 0, 0]
        self.cellTrans_options.rotate_flag = False
        self.cellTrans_options.phi = 0
        self.cellTrans_options.theta = 0
        self.cellTrans_options.psi = 0

        # opsin options
        self.opsin_options = Dict()
        # 'soma', 'all', 'apic' or 'apical', 'dend' or 'basal', 'alldend', 'axon', 'apicaltrunk', 'apicaltrunk_ext', 'apicaltuft', 'obliques', [specific list]
        self.opsin_options.opsinlocations = 'soma'
        self.opsin_options.opsinmech = 'chr2h134r'
        # uS if none then not included in calculation if value => gbar are rescaled so total G is Gmax_total
        self.opsin_options.Gmax_total = None
        # directly make connection with xtra mechanism upon opsin insertion
        self.opsin_options.set_pointer_xtra = True
        # example of uniform #distribution = lambda seg_xyz: 5*rv.pdf(np.linalg.norm(np.array(seg_xyz)-source))/rv.pdf(0) with source [x,y,z] and rv is truncnorm from scipy
        self.opsin_options.distribution = lambda x: 10
        self.opsin_options.distribution_method = '3d'  # '3d' or 'hdistance'
        # or e.g. 'soma[0](0.5)'
        self.opsin_options.distribution_source_hdistance = None

        '''
        # example for truncated norm based on 3D positions
        # for path dependance 'np.linalg.norm(np.array(seg_xyz)-source)' -> x
        from scipy.stats import truncnorm
        scale = 200
        a, b = (-1000 - 0)/scale, (1000 - 0)/scale
        source = [300,300,300]
        rv = truncnorm(a,b,0,scale)
        distribution = lambda x: 5*rv.pdf(np.linalg.norm(np.array(x)-source))/rv.pdf(0)
        with x: seg_[x,y,z]
        '''

    def setParam(self, label, param, value):
        if label in self:
            d = self[label]
        else:
            return False

        d[param] = value

        return True

    def rename(self, old, new, label=None):
        return self.__rename__(old, new, label)


class stimOptions(Dict):
    def __init__(self):
        self._init_flag = True
        self.stim_type = [None]  # Optogxstim or eVstim

        # optical stimulation parameters
        self.Ostimparams = Dict()
        self.Ostimparams.field = None
        self.Ostimparams.method_prepareData = 'ninterp'
        self.Ostimparams.filepath = None
        self.Ostimparams.delay = 50
        self.Ostimparams.dur = 10
        self.Ostimparams.amp = 1
        self.Ostimparams.structured = True
        self.Ostimparams.pulseType = 'singleSquarePulse'  # or pulseTrain
        # options are 'prf', 'dc', phi, theta, psi, c0(point where to rotate around), xT (translation), fill_value, InterpolMethod
        self.Ostimparams.options = {}

        # electrical stimulation parameters
        self.Estimparams = Dict()
        self.Estimparams.field = None
        self.Estimparams.method_prepareData = 'ninterp'
        self.Estimparams.filepath = None
        self.Estimparams.delay = 0
        self.Estimparams.dur = 10
        self.Estimparams.amp = 1
        self.Estimparams.structured = True
        self.Estimparams.pulseType = 'singleSquarePulse'
        self.Estimparams.options = {}

        self._init_flag = False

    def get_dt_simintm(self, key, dt_init, fs=20):
        params = getattr(self, key)
        if params.pulseType == 'singleSquarePulse':
            dtlist = [dt_init, min(dt_init, params.dur/fs)]
            simtimelist = [params.delay, params.delay+params.dur*2]
        elif params.pulseType == 'pulseTrain':
            prf = params.options['prf']
            dc = params.options['dc']
            if dc > 0.5:
                print(
                    '!!!!!!!!!! piecewise dt adjustment only works for dc<=0.5, (dt determined based on on time)!!!!!!!!')
            spd = dc/prf  # single pulse duration
            dt = min(dt_init, spd/fs)

            dtlist = [dt, dt_init]
            simtimelist = [spd*2, 1/prf]
            while simtimelist[-1] < params.dur:
                dtlist.extend([dt, dt_init])
                simtimelist.extend([simtimelist[-1]+x for x in [spd*2, 1/prf]])
            simtimelist = [params.delay +
                           x for x in simtimelist if x <= params.dur]
            dtlist = [dt_init]+dtlist[:len(simtimelist)]
            simtimelist = [params.delay]+simtimelist
        else:
            raise NotImplementedError

        return dtlist, simtimelist

    def __setattr__(self, key, value):
        # new setattr because we want stim_type to be always converted to array
        # only check if not in initialization
        if key != '_init_flag':
            if self._init_flag:
                super().__setattr__(key, value)
            else:
                try:
                    # if attribute wasn't there do not create
                    self.__getattr__(key)
                    if key == 'stim_type':
                        if not isinstance(value, list):
                            value = [value]

                except AttributeError:
                    raise AttributeError(key)

                else:
                    super().__setattr__(key, value)
        else:
            object.__setattr__(self, key, value)

    def setParam(self, label, param, value):
        if label in self:
            d = self[label]
        else:
            return False

        d[param] = value

        return True

    def rename(self, old, new, label=None):
        return self.__rename__(old, new, label)


class analysesOptions(Dict):
    """
    Class to hold options form Model/Cells

    """

    def __init__(self):
        self.sec_plot_flag = True
        self.print_secpos = False

        self.recordTotalOptogeneticCurrent = True
        self.recordSuccesRatio = True
        self.succesRatioOptions = Dict({'type': ['eVstim', 'Optogenx'], 'window': [
                                       10], 'succesRatio_seg': 'soma[0](0.5)'})

        self.shapeplots = [dict(cvals_type='os', colorscale='log10'), dict(
            cvals_type='gchr2bar_chr2h134r')]
        self.shapeplot_axsettings = {}
        self.save_shapeplots = True
        self.shapeplots_extension = '.png'

        self.recordTraces = Dict({'v': dict(sec='all', loc=0.5, var='v'), 'ichr2': dict(
            sec='all', loc=0.5, var='i', mech='chr2h134r')})
        self.tracesplot_axsettings = {}
        self.save_traces = True
        self.traces_extension = '.png'
        self.samplefrequency_traces = 5000  # Hz

        self.recordAPs = 'all'  # 'all', 'all0.5' all -> all segments all0.5 only centre segment
        self.apthresh = -10
        self.preordersecforplot = True
        self.save_rasterplot = True
        self.rasterplotopt = {}

        self.save_SDplot = True
        self.SDeVstim = Dict()
        self.SDeVstim.durs = np.logspace(-3, 3, 7)
        self.SDeVstim.startamp = 1
        self.SDeVstim.cellrecloc = 'soma[0](0.5)'
        self.SDeVstim.stimtype = 'cathodic'
        self.SDeVstim.stimpointer = 'estim_xtra'
        self.SDeVstim.nr_pulseOI = 1
        self.SDeVstim.return_spikeCountperPulse = True
        self.SDeVstim.options = dict(
            vinit=-65, simdur=400, dc_sdc=0.5, n_iters=7)

        self.SDOptogenx = Dict()
        self.SDOptogenx.durs = np.logspace(-3, 3, 7)
        self.SDOptogenx.startamp = 1000
        self.SDOptogenx.cellrecloc = 'soma[0](0.5)'
        self.SDOptogenx.stimtype = 'optogenx'
        self.SDOptogenx.stimpointer = 'ostim_xtra'
        self.SDOptogenx.nr_pulseOI = 1
        self.SDOptogenx.return_spikeCountperPulse = True
        self.SDOptogenx.record_iOptogenx = None  # opsin mechanisme e.g., 'chr2h134r'
        # for scale_initsearch = 10 error is 1/(2**7)
        self.SDOptogenx.options = dict(
            vinit=-65, simdur=400, dc_sdc=0.5, n_iters=7)

        self.save_VTAplot = True
        self.VTAeVstim = Dict()
        self.VTAeVstim.startpos = np.array(
            [np.zeros(5), np.zeros(5), np.arange(0, 1, 0.2)+0.2]).T
        self.VTAeVstim.searchdir = np.array([1, 0, 0])
        self.VTAeVstim.cellrecloc = 'soma[0](0.5)'
        self.VTAeVstim.stimtype = 'cathodic'
        self.VTAeVstim.stimpointer = 'estim_xtra'
        self.VTAeVstim.nr_pulseOI = 1
        # for scale_initsearch = 10 error is 1/(2**7)
        self.VTAeVstim.options = dict(
            vinit=-65, simdur=400, dc_sdc=0.5, n_iters=7)

        self.VTAOptogenx = Dict()
        self.VTAOptogenx.startpos = np.array(
            [np.zeros(5), np.zeros(5), np.arange(0, 1, 0.2)+0.2]).T
        self.VTAOptogenx.searchdir = np.array([1, 0, 0])
        self.VTAOptogenx.cellrecloc = 'soma[0](0.5)'
        self.VTAOptogenx.stimtype = 'optogenx'
        self.VTAOptogenx.stimpointer = 'ostim_xtra'
        self.VTAOptogenx.nr_pulseOI = 1
        # for scale_initsearch = 10 error is 1/(2**7)
        self.VTAOptogenx.options = dict(
            vinit=-65, simdur=400, dc_sdc=0.5, n_iters=7)

    def setParam(self, label, param, value):
        if label in self:
            d = self[label]
        else:
            return False

        d[param] = value

        return True

    def rename(self, old, new, label=None):
        return self.__rename__(old, new, label)


def replaceNeuronSectionsandFunTostr(obj):
    if type(obj) == list:
        for i, item in enumerate(obj):
            if type(item) in [list, dict]:
                item = replaceNeuronSectionsandFunTostr(item)
            elif type(item) == neuron.nrn.Section or type(item) == neuron.nrn.Segment:
                obj[i] = str(item)
            elif callable(item):
                return inspect.getsource(item)

    elif type(obj) == dict:
        for key, val in obj.items():
            if type(val) in [list, dict]:
                obj[key] = replaceNeuronSectionsandFunTostr(val)
            elif type(val) == neuron.nrn.Section or type(val) == neuron.nrn.Segment:
                obj[key] = str(val)
            elif callable(val):
                obj[key] = inspect.getsource(val)
    elif type(obj) == neuron.nrn.Section or type(obj) == neuron.nrn.Segment:
        return str(obj)
    elif callable(obj):
        return inspect.getsource(obj)

    return obj


def recursiveDictUpdate(obj, value, checkhasattr_flag=True):
    for k, v in value.items():
        # add in brackets parameter names that do not need to be checked typically dict with variable keys if dict with dict here else add on next line
        if isinstance(v, (dict, Dict)) and (not k in ['options']):
            if len(v) == 0:
                setattr(obj, k, Dict(v))
            else:
                if k in []:
                    recursiveDictUpdate(getattr(obj, k), Dict(
                        v), checkhasattr_flag=False)
                else:
                    recursiveDictUpdate(getattr(obj, k), Dict(
                        v), checkhasattr_flag=True)
        else:
            if v == 'removedToReduceMemory':
                v = None
            # elif isinstance(v,str) and any([x in v for x in ['= lambda','=lambda']]):
                # convert str(functions) back to functions/callables
            #    v = eval(v.split('=')[-1])
            if checkhasattr_flag:
                if hasattr(obj, k):
                    setattr(obj, k, v)
                else:
                    raise AttributeError(k)
            else:
                setattr(obj, k, v)
