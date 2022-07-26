from .globalFunctions.utils import Dict, replaceDictODict
import numpy as np
from datetime import datetime

class simParams(object):
    def __init__(self, simParamsDict = None):
        # Flags
        self.test_flag = False
        self.plot_flag = False
        self.autosetPointer = True

        self.seed = datetime(2021, 12, 1, 13, 0).timestamp()

        self.duration = 100
        self.samplingFrequency = 10000 #Hz
        self.dt = 0.025
        self.celsius = 20
        self.defaultThreshold = -10
        self.v0 = -65

        self.resultsFolder = '/Temp'
        self.subfolderSuffix = ''
        self.signif = 4

        self.cellsopt = cellsOptions()
        self.stimopt = stimOptions()
        self.analysesopt = analysesOptions()

        if simParamsDict:
            #overwrite defaults
            netParamsComponents = ['cellsopt', 'stimopt','analysesopt']
            for k,v in simParamsDict.items():
                if k in netParamsComponents:
                    recursiveDictUpdate(getattr(self, k),v)
                else:
                    if hasattr(self,k):
                        if isinstance(v, dict):
                            setattr(self, k, Dict(v))
                        else:
                            setattr(self, k, v)
                    else:
                        raise AttributeError(k)

    def todict(self,reduce,inplace = False):
        if not inplace:
            out = replaceDictODict(self.__dict__).copy()
        else:
            out = replaceDictODict(self.__dict__)
        if reduce:
            msg = 'removedToReduceMemory'
            # add here variables that need to be substituted by msg

        return out

class cellsOptions(Dict):
    """
    Class to hold options form Model/Cells

    """
    def __init__(self):
        self.neurontemplate = 'CA1_PC_cAC_sig5' # for other options see cells.Neurontemplates
        self.extracellular = False   # include extracellular mechinism for extracellular electrical stimulation

        # initialization options
        # options of the neurontemplate class
        self.init_options = Dict()
        self.init_options.replace_axon = True
        self.init_options.morphologylocation ='./Model/morphologies'
        self.init_options.ID=0
        self.init_optionsty=0
        self.init_options.col=0
        self.init_options.phi = 0
        self.init_options.theta = 0
        self.init_options.movesomatoorigin = True

        # cell transformation options
        # !Note!: currently in code first rotation then movement (with init_options movesomatoorigin true soma is at origin)
        self.cellTrans_options = Dict()
        self.cellTrans_options.move_flag = False
        self.cellTrans_options.rt = [0,0,0]
        self.cellTrans_options.rotate_flag = False
        self.cellTrans_options.phi = 0
        self.cellTrans_options.theta = 0

        # opsin options
        self.opsin_options = Dict()
        self.opsin_options.opsinlocations = 'soma' #'soma', 'all', 'apic' or 'apical', 'dend' or 'basal', 'alldend', 'axon', 'apicaltrunk', 'apicaltrunk_ext', 'apicaltuft', 'obliques', [specific list]
        self.opsin_options.opsinmech = 'chr2h134r'
        self.opsin_options.set_pointer_xtra = True # directly make connection with xtra mechanism upon opsin insertion
        self.opsin_options.distribution = lambda x: 10 # example of uniform
        self.opsin_options.distribution_method = '3d'  # '3d' or 'hdistance'
        self.opsin_options.distribution_source_hdistance = None # or e.g. 'soma[0](0.5)'

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
        self.Ostimparams.method_prepareData='ninterp'
        self.Ostimparams.filepath= None
        self.Ostimparams.delay = 50
        self.Ostimparams.dur = 10
        self.Ostimparams.amp = 1
        self.Ostimparams.structured = True
        self.Ostimparams.pulseType =  'singleSquarePulse'
        self.Ostimparams.options =  {} # options are 'prf', 'dc', phi, theta, psi, c0(point where to rotate around), xT (translation), fill_value, InterpolMethod


        # electrical stimulation parameters
        self.Estimparams = Dict()
        self.Estimparams.field = None
        self.Estimparams.method_prepareData='ninterp'
        self.Estimparams.filepath= None
        self.Estimparams.delay = 0
        self.Estimparams.dur = 10
        self.Estimparams.amp = 1
        self.Estimparams.structured = True
        self.Estimparams.pulseType =  'singleSquarePulse'
        self.Estimparams.options =  {}

        self._init_flag = False
    def __setattr__(self, key, value):
        #new setattr because we want stim_type to be always converted to array
        #only check if not in initialization
        if key!='_init_flag':
            if self._init_flag:
                super().__setattr__(key, value)
            else:
                try:
                    #if attribute wasn't there do not create
                    self.__getattr__(key)
                    if key=='stim_type':
                        if not isinstance(value,list):
                            value = [value]

                except AttributeError:
                    raise AttributeError(key)

                else:
                    super().__setattr__(key, value)
        else:
            object.__setattr__(self,key, value)


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
        self.shapeplots = [dict(cvals_type='os',colorscale='log10'),dict(cvals_type='gchr2bar_chr2h134r')]
        self.sec_plot_flag = True
        self.print_secpos = True
        self.recordTraces = {'v':dict(sec='all', loc = 0.5, var='v'),'ichr2':dict(sec='all', loc=0.5, var = 'i', mech='chr2h134r')}




    def setParam(self, label, param, value):
        if label in self:
            d = self[label]
        else:
            return False

        d[param] = value

        return True

    def rename(self, old, new, label=None):
        return self.__rename__(old, new, label)



def recursiveDictUpdate(obj,value,checkhasattr_flag=True):
    for k,v in value.items():
        if isinstance(v,(dict,Dict)) and (not k in []): # add in brackets parameter names that do not need to be checked typically dict with variable keys if dict with dict here else add on next line
            if k in []:
                recursiveDictUpdate(getattr(obj,k),Dict(v),checkhasattr_flag=False)
            else:
                recursiveDictUpdate(getattr(obj,k),Dict(v),checkhasattr_flag=True)
        else:
            if v == 'removedToReduceMemory':
                v = None
            if checkhasattr_flag:
                if hasattr(obj,k):
                    setattr(obj,k,v)
                else:
                    raise AttributeError(k)
            else:
                setattr(obj,k,v)