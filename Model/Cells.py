from neuron import h
import numpy as np
import re, mmap

if __name__ == '__main__':
    import os, sys
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except:
        SCRIPT_DIR = os.path.dirname(os.getcwd())
    sys.path.append(SCRIPT_DIR)
    sys.path.append(os.path.dirname(SCRIPT_DIR))
    sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    print(f"add paths: {SCRIPT_DIR}, {os.path.dirname(SCRIPT_DIR)} and {(os.path.dirname(os.path.dirname(SCRIPT_DIR)))} to path")

from Functions.globalFunctions.utils import _Rx, _Ry, _Rz
import Functions.globalFunctions.morphology_v2 as mphv2
import matplotlib.pyplot as plt
import pandas as pd
import Functions.globalFunctions.utils as utils
from matplotlib import cm
from matplotlib import use as mpluse

global hShape_flag
if not 'hShape_flag' in globals():
    hShape_flag = False

morphocelltype ={
    '990803': 'SP_PC','050921AM2':'SP_PC','mpg141017_a1-2_idC':'SP_PC', 'mpg141208_B_idA':'SP_PC', 'mpg141209_A_idA':'SP_PC', 'mpg141209_B_idA':'SP_PC','mpg141215_A_idA':'SP_PC','mpg141216_A_idA':'SP_PC','mpg141217_A_idB':'SP_PC','mpg150305_A_idB':'SP_PC','oh140521_B0_Rat_idA':'SP_PC','oh140521_B0_Rat_idC':'SP_PC','oh140807_A0_idA':'SP_PC','oh140807_A0_idB':'SP_PC','oh140807_A0_idC':'SP_PC','oh140807_A0_idF':'SP_PC','oh140807_A0_idG':'SP_PC','oh140807_A0_idH':'SP_PC','oh140807_A0_idJ':'SP_PC','010710HP2': 'SP_Ivy','011017HP2': 'SO_OLM','011023HP2': 'SO_BS','011127HP1': 'SLM_PPA','031031AM1': 'SP_CCKBC','060314AM2': 'SP_PVBC','970509HP2': 'SO_Tri','970627BHP1': 'SP_PVBC','970717D': 'SP_Ivy','970911C': 'SP_AA','971114B': 'SO_Tri','980120A': 'SO_BP','980513B': 'SP_BS','990111HP2': 'SP_PVBC','990611HP2': 'SR_SCA','990827IN5HP3': 'SR_IS1',
}
NeuronTemplates = ['CA1_PC_cAC_sig5','CA1_PC_cAC_sig6','bACnoljp8','bACnoljp7','cNACnoljp1','cNACnoljp2','INT_cAC_noljp4','INT_cAC_noljp3']

class NeuronTemplate:
    '''
    Parent class for loading a neuron specified in hoc file.
    Adds metadata and contains multiple methods to manipulate cell.
    Required inputs are:
    * templatepath: cell template path (path to .hoc file)
    * templatename: cell name (template name given in .hoc file)
    Additional inputs are:
    * replace_axon = True: replace axon by stub axon (see .hoc files)
    * morphologylocation = ./Model/morphologies:  directory where neuron morphologies are stored
    * ID = 0
    * ty = 0
    * col = 0
    * phi = 0: initial rotation around z-axis (call rotate_Cell method with init_rotation true to apply rotation)
    * theta = 0: initial rotation around the y-axis (call rotate_Cell method with init_rotation true to apply rotation) order is first around Rz*Ry*r
    * movesomatoorigin = True: flag that is used in all child classes
    * **kwargs not used: add so no error when transfering kwargs from child classes
    '''
    def __init__ (self,templatepath, templatename, replace_axon = True, morphologylocation = './Model/morphologies',ID=0,ty=0,col=0, phi = 0, theta = 0, psi=0, movesomatoorigin = True,**kwargs):
        self.templatepath = templatepath
        self.templatename = templatename
        self.morphologylocation = morphologylocation
        self.replace_axon = str(replace_axon).lower()
        self.celsius = 34
        self.ID=ID
        self.ty=ty
        self.col=col
        self.phi = phi
        self.theta = theta
        self.psi = psi
        self.movesomatoorigin = movesomatoorigin

    def load_template(self):
        self.assign_MorphandCelltype() # read morphology and add corresponding cell type
        h.load_file(self.templatepath) # Load cell info
        self.template = getattr(h,self.templatename)(self.replace_axon,self.morphologylocation) # initialize cell
        try:
            # try to create allsec list (used in many methods)
            self.allsec = []
            for x in self.template.all:
                self.allsec.append(x)
        except Exception as e:
            print(e)

    def move_attributes(self):
        # cell is loaded under template attribute -> move level up
        for x in self.template.__dir__():
            if x != '__class__':
                setattr(self,x,getattr(self.template,x))
        self.template = None

    def move_Cell(self,rt):
        # move Cell with translate vector rt
        for section in self.allsec:
            for i in range(section.n3d()):
                xyz = np.array([section.x3d(i),section.y3d(i),section.z3d(i)])
                xyz = xyz+rt
                h.pt3dchange(i, xyz[0], xyz[1], xyz[2], section.diam3d(i), sec=section)

    def allign_cell_toaxis(self,axis = 'x'):
        '''
        Allign cell to axis = x,y or z
        For generallity -> extract principal component of a cell. For pyramidal cell this is probably the axo-somato-dendritic-axis.
        Allign this axis with the chosen euler axis.
        '''
        # determine principal component axis assuming this is the axo-somato-dendritic axis
        secpos = self.gather_secpos()
        #secpos = secpos-secpos.mean()
        cov_mat = secpos.cov()
        eig_values, eig_vectors = np.linalg.eig(cov_mat)
        e_indices = np.argmax(eig_values)
        # principal component axis: p1
        p1 = eig_vectors[:,e_indices]
        # spherical angles
        phi = np.arctan2(p1[1],p1[0])
        try:
            na = self.apic[-1].n3d()-1
            xyz = np.array([self.apic[-1].x3d(na),self.apic[-1].y3d(na),self.apic[-1].z3d(na)])
            angle = np.arccos(np.dot(xyz/np.linalg.norm(xyz),p1))
            phi = phi+np.pi if angle>np.pi/2 else phi
        except:
            phi = phi+np.pi if phi<0 else phi #most of time morphology files are built with apical dendrites to the positive y-direction. correct if p1 is in negative y direction
        theta = np.arccos(p1[2])
        # Rotate to axis
        self.rotate_Cell(phi=phi,theta=np.pi/2-theta,psi=0,inverse=True)

        if axis=='y':
            self.rotate_Cell(phi=np.pi/2,theta=0,psi=0)
        elif axis == 'z':
            self.rotate_Cell(phi=0,theta=-np.pi/2,psi=0)
        elif axis != 'x':
            raise ValueError('wrong axis value given. Should be x,y or z')


    def rotate_Cell(self,phi=0,theta=0,psi=0,inverse = False, init_rotation=False, allign_axis = True):
        '''
        rotate cell Tait-Bryan angles Rz(phi)*Ry(theta)*Rx(psi)*r
        phi = yawn, theta=pitch, psi = roll
        init_rotation -> use internal phi and theta (self.phi, self theta, self.psi)
        allign_axis -> first allign principal (component) axis (often axo-somato-dendritic) to x-axis -> phi and theta will be spherical coordinates of the axis
        inverse: -> do inverse rotation R = R.T
        phi, theta and psi are zero by default
        '''
        if init_rotation:
            phi = self.phi
            theta = self.theta
            psi = self.psi
            if allign_axis:
                # allign principal axis to x-axis. (principal axis is first principal component from 3D points)
                self.allign_cell_toaxis()
        if phi!=0 or theta!=0 or psi!=0:
            for section in self.allsec:
                for i in range(section.n3d()):
                    xyz = np.array([section.x3d(i),section.y3d(i),section.z3d(i)])
                    R = np.dot(_Rz(phi),np.dot(_Ry(theta),_Rx(psi)))
                    if inverse:
                        R = R.T
                    xyz = np.dot(R,xyz[:,None])
                    h.pt3dchange(i, xyz[0][0], xyz[1][0], xyz[2][0], section.diam3d(i), sec=section)

    def insertExtracellular(self, seclist='all', set_pointer_xtra=True):
        '''
        add extracellular mechanism to cell sections.
        * seclist = 'all': seclist can be provided, if all -> swap to self.allsec
        * set_pointer_xtra = True: create pointer between seg.extracellular._ref_e and seg.extra_ex
        '''
        if seclist == 'all':
            seclist = self.allsec
        for sec in seclist:
            sec.insert('extracellular')
            if not sec.has_membrane('xtra'):
                sec.insert('xtra')
            if set_pointer_xtra:
                for seg in sec:
                    h.setpointer(seg.extracellular._ref_e, 'ex', seg.xtra)

    def insertOptogenetics(self,seclist='all',opsinmech = 'chr2h134r', set_pointer_xtra=True):
        '''
        Insert opsin defined via opsinmech
        * opsinmech = 'chr2h134r'
        * seclist = 'all': seclist can be provided, if all -> swap to self.allsec
        * set_pointer_xtra = True: create pointer between seg.extracellular._ref_Iopto and seg.extra_ox
        '''
        if seclist == 'all':
            seclist = self.allsec
        for sec in seclist:
            sec.insert(opsinmech)
            if not sec.has_membrane('xtra'):
                sec.insert('xtra')
            if set_pointer_xtra:
                for seg in sec:
                    h.setpointer(getattr(seg,opsinmech)._ref_Iopto, 'ox', seg.xtra)

    def updateXtraCoors(self):
        # update stored location in xtra mechanism
        # xtra mechanism contains variables that facilitates access. For instance 3d position of a segment (based on aberra et al 2018 code on modelDB)
        for sec in self.allsec:
            if sec.has_membrane('xtra'):
                pt3ds = [[getattr(sec,x)(i) for i in range(sec.n3d())] for x in ['x3d', 'y3d', 'z3d']]
                arcl = np.array([sec.arc3d(i) for i in range(sec.n3d())])
                arcl = arcl/arcl[-1] #normalize
                for seg in sec:
                    xyz_seg = [np.interp(seg.x,arcl,coors) for coors in pt3ds]
                    seg.x_xtra, seg.y_xtra, seg.z_xtra = xyz_seg

    def assign_MorphandCelltype(self):
        # scans file for loaded morphology and assigns to metadate + add corresponding cell type
        #https://stackoverflow.com/questions/31019854/typeerror-cant-use-a-string-pattern-on-a-bytes-like-object-in-re-findall
        with open(self.templatepath, 'r') as f:
            data = mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ).read().decode("utf-8")
            p = re.compile(r'load_morphology\("morphologies", .*\)')
            self.morphology = re.findall(r' (".*[(.asc)]")',p.search(data).group(0))[0][1:-5]
        self.celltype = morphocelltype[self.morphology]

    def moveSomaToOrigin(self):
        if self.movesomatoorigin:
            # !!! only call hSahpe once. Do not know why but if called twice and neuron has moved -> then translates some sections twice!!!
            # h.Shape needs to be called to init 3d positions of sections
            global hShape_flag
            if not hShape_flag:
                h.Shape(False)
                hShape_flag = True
            soma_pos = np.array([[self.soma[0].x3d(i),self.soma[0].y3d(i),self.soma[0].z3d(i)] for i in range(self.soma[0].n3d())])
            soma_pos = np.mean(soma_pos,axis=0)
            self.move_Cell(-soma_pos)

    def gather_secpos(self,print_flag=False):
        # store center of each section in dataframe
        secpos = {}
        for sec in self.allsec:
            sec_name = str(sec).split('.')[-1]
            xyz = mphv2.get_section_path(h,sec)
            xyz = np.mean(xyz, axis=0)
            secpos[sec_name]=xyz
        secpos = pd.DataFrame(secpos).T.rename(lambda x:['x','y','z'][x], axis = 1)
        secpos = secpos.applymap(lambda x: utils.signif(x,2))
        if print_flag:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                print(secpos)
        return secpos

    def sec_plot(self,ax=None):
        if ax is None:
            # if no axes provided create new figure with 3d axes
            fig = plt.figure()
            ax = plt.subplot(111,projection='3d')
        global hShape_flag
        if not hShape_flag:
            # h.Shape needs to be called but only !!once!! otherwise problems with translation
            h.Shape(False)
            hShape_flag = True

        colorlist = []
        colorkeyval = {'soma':'tab:red', 'axon':'tomato','apical trunk':'tab:blue','apical trunk ext':'royalblue', 'apical tuft': 'tab:green','apical obliques': 'tab:cyan', 'basal dendrites': 'tab:olive', 'unclassified':[0,0,0]}
        for sec in self.allsec:
            for seg in sec:
                if 'soma' in str(sec):
                    colorlist.append(colorkeyval['soma'])
                elif 'axon' in str(sec):
                    colorlist.append(colorkeyval['axon'])
                elif sec in self.apicalTrunk:
                    colorlist.append(colorkeyval['apical trunk'])
                elif sec in self.apicalTrunk_ext:
                    colorlist.append(colorkeyval['apical trunk ext'])
                elif sec in self.apicalTuft:
                    colorlist.append(colorkeyval['apical tuft'])
                elif sec in self.apical_obliques:
                    colorlist.append(colorkeyval['apical obliques'])
                elif 'dend' in str(sec):
                    colorlist.append(colorkeyval['basal dendrites'])
                else:
                    colorlist.append([0,0,0])
        mphv2.shapeplot(h,ax,sections = self.allsec,cvals = colorlist,cb_flag=False,clim=[0,0])
        ax.set_title(f"{self.templatename} / {self.morphology} / {self.celltype}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.grid(visible=None)
        ax.axis('on')
        for k,v in colorkeyval.items():
            ax.plot(np.nan,np.nan,np.nan,label=k,color=v)
        ax.legend(frameon=False)
        #ax.view_init(elev=0, azim=0)
        return ax

    def check_pointers(self,autoset = False):
        for sec in self.allsec:
            if sec.has_membrane('xtra'):
                for seg in sec:
                    for x,y in zip(['ox','ex'],['os','es']):
                        try:
                            getattr(seg.xtra,x)
                        except Exception as E:
                            if autoset:
                                if 'was not made to point to anything' in E.args[0]:
                                    h.setpointer(getattr(seg.xtra,'_ref_'+y), x, seg.xtra)
                                else:
                                    print(E)
                            else:
                                raise E

    def updateMechValue(self,seglist,values,mech):
        segwvalue = []
        segwovalue = []
        if seglist=='all':
            seglist = [seg for sec in self.allsec for seg in sec]
        if isinstance(values,(float,int)):
            values = len(seglist)*[values]
        if len(seglist)!=len(values):
            raise ValueError('seglist and values need to be same length or values needs to be float or integer')
        for seg, value in zip(seglist,values):
            if hasattr(seg,mech.rsplit('_',1)[-1]):
                setattr(seg,mech,value)
                segwvalue.append((str(seg),value))
            else:
                segwovalue.append(str(seg))

        return segwvalue,segwovalue

    def distribute_mechvalue(self,distribution,seclist='all',method='3d', source_hdistance = None):
        # method: '3d', 'hdistance' 3d positions or via h.distance
        # if hdistance selected use h.distance method from neuron (distance along path between two segments)
        # alse source_hdistance needs to be provided. i.e., a cell segment
        values = []
        seglist = []
        if isinstance(distribution,str) and any([x in distribution for x in ['= lambda','=lambda']]):
            #convert str(functions) back to functions/callables
            distribution = eval(distribution.split('=')[-1])
        if seclist=='all':
            seclist = self.allsec
        if method.lower()=='3d':
            for sec in seclist:
                pt3ds = [[getattr(sec,x)(i) for i in range(sec.n3d())] for x in ['x3d', 'y3d', 'z3d']]
                arcl = np.array([sec.arc3d(i) for i in range(sec.n3d())])
                arcl = arcl/arcl[-1] #normalize
                for seg in sec:
                    xyz_seg = [np.interp(seg.x,arcl,coors) for coors in pt3ds]
                    values.append(distribution(xyz_seg))
                    seglist.append(seg)
        elif method.lower()=='hdistance':
            if source_hdistance is not None:
                for sec in seclist:
                    for seg in sec:
                        values.append(distribution(h.distance(source_hdistance,seg)))
                        seglist.append(seg)
            else:
                raise ValueError('define source_hdistance')
        else:
            raise ValueError('possible methods are 3d or hdistance')
        return seglist, values
    def calc_Gmax_mechvalue(self, mech,values = None, seglist = 'all'):
        conv = 1e-5 #conversion factor assume value is in mS/cm2 area in um2 -> G in uS
        if seglist=='all':
            seglist = [seg  for sec in self.allsec for seg in sec if hasattr(seg,mech)]
        if values is None:
            values = [getattr(seg,mech) for seg in seglist]
        # trim lists
        if len(seglist)!=len(values):
            raise ValueError('seglist and values should have same length')

        seglist,values = map(lambda x: list(x),zip(*filter(lambda x: hasattr(x[0],mech), zip(seglist,values))))
        G = sum([seg.area()*val*conv for seg,val in zip(seglist,values)])
        return G, seglist, values


class CA1_PC_cAC_sig5(NeuronTemplate):
    '''
    CA1 pyramidal cell
    loads './Model/cell_seed4_0-pyr-04.hoc' with templatename = 'CA1_PC_cAC_sig5'
    '''
    def __init__ (self,**kwargs):
        super().__init__(templatepath = './Model/cell_seed4_0-pyr-04.hoc', templatename = 'CA1_PC_cAC_sig5',**kwargs) #init of parent class
        self.load_template()
        self.move_attributes()
        self.make_lists()
        self.moveSomaToOrigin()
        self.rotate_Cell(init_rotation=True)
    def __str__ (self):
        try:
            return f'compartCell_{self.__class__.__name__}_{self.ID}'
        except: return 'compartCell%d'%self.ID
    def __repr__ (self):
        return self.__str__()
    def make_lists(self):
        # craeate some specific list of section -> facilitates access
        self.allsec = []
        self.alldend = []
        self.apicalTrunk = []
        apicalTrunk = ['apic[0]','apic[6]','apic[8]','apic[10]','apic[12]','apic[18]','apic[20]','apic[24]','apic[26]','apic[30]','apic[46]','apic[60]','apic[62]','apic[66]','apic[68]','apic[70]']

        self.apicalTrunk_ext = []
        apicalTrunk_ext = apicalTrunk+['apic[31]','apic[33]','apic[39]','apic[47]','apic[51]']

        self.apicalTuft = []
        apicalTuft = ['apic[40]','apic[41]','apic[42]','apic[43]','apic[44]','apic[45]','apic[52]','apic[53]','apic[54]','apic[55]','apic[56]','apic[57]','apic[58]','apic[59]','apic[71]','apic[72]','apic[73]','apic[74]','apic[75]','apic[76]','apic[77]','apic[78]','apic[79]','apic[80]','apic[81]','apic[82]','apic[83]','apic[84]','apic[85]','apic[86]','apic[87]','apic[88]']

        self.apical_obliques = []
        obliques = ['apic[1]','apic[2]','apic[3]','apic[4]','apic[5]','apic[7]','apic[9]','apic[11]','apic[13]','apic[14]','apic[15]','apic[16]','apic[17]','apic[19]','apic[21]','apic[22]','apic[23]','apic[25]','apic[27]','apic[28]','apic[29]','apic[32]','apic[34]','apic[35]','apic[36]','apic[37]','apic[38]','apic[48]','apic[49]','apic[50]','apic[61]','apic[63]','apic[64]','apic[65]','apic[67]','apic[69]']

        for x in self.all:
            self.allsec.append(x)
            if (not 'soma' in str(x)) and (not 'axon' in str(x)):
                self.alldend.append(x)
            if any([y in str(x) for y in apicalTrunk]):
                self.apicalTrunk.append(x)
            if any([y in str(x) for y in apicalTrunk_ext]):
                self.apicalTrunk_ext.append(x)
            if any([y in str(x) for y in apicalTuft]):
                self.apicalTuft.append(x)
            if any([y in str(x) for y in obliques]):
                self.apical_obliques.append(x)

class CA1_PC_cAC_sig6(NeuronTemplate):
    def __init__ (self, **kwargs):
        super().__init__(templatepath = './Model/cell_seed3_0-pyr-08.hoc', templatename = 'CA1_PC_cAC_sig6',**kwargs)
        self.load_template()
        self.move_attributes()
        self.make_lists()
        self.moveSomaToOrigin()
        self.rotate_Cell(init_rotation=True)
    def __str__ (self):
        try:
            return f'compartCell_{self.__class__.__name__}_{self.ID}'
        except: return 'compartCell%d'%self.ID
    def __repr__ (self):
        return self.__str__()
    def make_lists(self):
        self.allsec = []
        self.alldend = []
        self.apicalTrunk = []
        apicalTrunk = [f'apic[{i}]' for i in [0,8,9,11,13,19,21,23,27,31,35]]

        self.apicalTrunk_ext = []
        apicalTrunk_ext = apicalTrunk+[f'apic[{i}]' for i in [36,40,53,61]]

        self.apicalTuft = []
        apicalTuft = [f'apic[{i}]' for i in [37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,54,55,56,57,58,59,60,61,62,63,64,65,66,67]]

        self.apical_obliques = []
        obliques = [f'apic[{i}]' for i in [1,2,3,4,5,6,7,10,12,14,15,16,17,18,20,22,24,25,26,28,29,30,32,33,34,68,69,70,71,72,73,74,75,76,77,78]]

        for x in self.all:
            self.allsec.append(x)
            if (not 'soma' in str(x)) and (not 'axon' in str(x)):
                self.alldend.append(x)
            if any([y in str(x) for y in apicalTrunk]):
                self.apicalTrunk.append(x)
            if any([y in str(x) for y in apicalTrunk_ext]):
                self.apicalTrunk_ext.append(x)
            if any([y in str(x) for y in apicalTuft]):
                self.apicalTuft.append(x)
            if any([y in str(x) for y in obliques]):
                self.apical_obliques.append(x)

class bACnoljp8(NeuronTemplate):
    def __init__ (self,**kwargs):
        super().__init__(templatepath = './Model/cell_seed1_0-bac-10.hoc', templatename = 'bACnoljp8',**kwargs)
        self.load_template()
        self.move_attributes()
        self.make_lists()
        self.moveSomaToOrigin()
        self.rotate_Cell(init_rotation=True)
    def __str__ (self):
        try:
            return f'compartCell_{self.__class__.__name__}_{self.ID}'
        except: return 'compartCell%d'%self.ID
    def __repr__ (self):
        return self.__str__()
    def make_lists(self):
        self.allsec = []
        self.alldend = []
        self.apicalTrunk = []
        apicalTrunk = [f'apic[{i}]' for i in [0,8,9,11,13,19,21,23,27,31,35]]

        self.apicalTrunk_ext = []
        apicalTrunk_ext = apicalTrunk+[f'apic[{i}]' for i in [36,40,53,61]]

        self.apicalTuft = []
        apicalTuft = [f'apic[{i}]' for i in [37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,54,55,56,57,58,59,60,61,62,63,64,65,66,67]]

        self.apical_obliques = []
        obliques = [f'apic[{i}]' for i in [1,2,3,4,5,6,7,10,12,14,15,16,17,18,20,22,24,25,26,28,29,30,32,33,34,68,69,70,71,72,73,74,75,76,77,78]]

        for x in self.all:
            self.allsec.append(x)
            if (not 'soma' in str(x)) and (not 'axon' in str(x)):
                self.alldend.append(x)
            if any([y in str(x) for y in apicalTrunk]):
                self.apicalTrunk.append(x)
            if any([y in str(x) for y in apicalTrunk_ext]):
                self.apicalTrunk_ext.append(x)
            if any([y in str(x) for y in apicalTuft]):
                self.apicalTuft.append(x)
            if any([y in str(x) for y in obliques]):
                self.apical_obliques.append(x)

class cNACnoljp1(NeuronTemplate):
    def __init__ (self,**kwargs):
        super().__init__(templatepath = './Model/cell_seed1_0-cnac-04.hoc', templatename = 'cNACnoljp1',**kwargs)
        self.load_template()
        self.move_attributes()
        self.make_lists()
        self.moveSomaToOrigin()
        self.rotate_Cell(init_rotation=True)
    def __str__ (self):
        try:
            return f'compartCell_{self.__class__.__name__}_{self.ID}'
        except: return 'compartCell%d'%self.ID
    def __repr__ (self):
        return self.__str__()
    def make_lists(self):
        self.allsec = []
        self.alldend = []
        self.apicalTrunk = []
        apicalTrunk = [f'apic[{i}]' for i in [0,8,9,11,13,19,21,23,27,31,35]]

        self.apicalTrunk_ext = []
        apicalTrunk_ext = apicalTrunk+[f'apic[{i}]' for i in [36,40,53,61]]

        self.apicalTuft = []
        apicalTuft = [f'apic[{i}]' for i in [37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,54,55,56,57,58,59,60,61,62,63,64,65,66,67]]

        self.apical_obliques = []
        obliques = [f'apic[{i}]' for i in [1,2,3,4,5,6,7,10,12,14,15,16,17,18,20,22,24,25,26,28,29,30,32,33,34,68,69,70,71,72,73,74,75,76,77,78]]

        for x in self.all:
            self.allsec.append(x)
            if (not 'soma' in str(x)) and (not 'axon' in str(x)):
                self.alldend.append(x)
            if any([y in str(x) for y in apicalTrunk]):
                self.apicalTrunk.append(x)
            if any([y in str(x) for y in apicalTrunk_ext]):
                self.apicalTrunk_ext.append(x)
            if any([y in str(x) for y in apicalTuft]):
                self.apicalTuft.append(x)
            if any([y in str(x) for y in obliques]):
                self.apical_obliques.append(x)

class bACnoljp7(NeuronTemplate):
    def __init__ (self,**kwargs):
        super().__init__(templatepath = './Model/cell_seed2_0-bac-06.hoc', templatename = 'bACnoljp7',**kwargs)
        self.load_template()
        self.move_attributes()
        self.make_lists()
        self.moveSomaToOrigin()
        self.rotate_Cell(init_rotation=True)
    def __str__ (self):
        try:
            return f'compartCell_{self.__class__.__name__}_{self.ID}'
        except: return 'compartCell%d'%self.ID
    def __repr__ (self):
        return self.__str__()
    def make_lists(self):
        self.allsec = []
        self.alldend = []
        self.apicalTrunk = []
        apicalTrunk = [f'apic[{i}]' for i in [0,8,9,11,13,19,21,23,27,31,35]]

        self.apicalTrunk_ext = []
        apicalTrunk_ext = apicalTrunk+[f'apic[{i}]' for i in [36,40,53,61]]

        self.apicalTuft = []
        apicalTuft = [f'apic[{i}]' for i in [37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,54,55,56,57,58,59,60,61,62,63,64,65,66,67]]

        self.apical_obliques = []
        obliques = [f'apic[{i}]' for i in [1,2,3,4,5,6,7,10,12,14,15,16,17,18,20,22,24,25,26,28,29,30,32,33,34,68,69,70,71,72,73,74,75,76,77,78]]

        for x in self.all:
            self.allsec.append(x)
            if (not 'soma' in str(x)) and (not 'axon' in str(x)):
                self.alldend.append(x)
            if any([y in str(x) for y in apicalTrunk]):
                self.apicalTrunk.append(x)
            if any([y in str(x) for y in apicalTrunk_ext]):
                self.apicalTrunk_ext.append(x)
            if any([y in str(x) for y in apicalTuft]):
                self.apicalTuft.append(x)
            if any([y in str(x) for y in obliques]):
                self.apical_obliques.append(x)

class cNACnoljp2(NeuronTemplate):
    def __init__ (self,**kwargs):
        super().__init__(templatepath = './Model/cell_seed2_0-cnac-08.hoc', templatename = 'cNACnoljp2',**kwargs)
        self.load_template()
        self.move_attributes()
        self.make_lists()
        self.moveSomaToOrigin()
        self.rotate_Cell(init_rotation=True)
    def __str__ (self):
        try:
            return f'compartCell_{self.__class__.__name__}_{self.ID}'
        except: return 'compartCell%d'%self.ID
    def __repr__ (self):
        return self.__str__()
    def make_lists(self):
        self.allsec = []
        self.alldend = []
        self.apicalTrunk = []
        apicalTrunk = [f'apic[{i}]' for i in [0,8,9,11,13,19,21,23,27,31,35]]

        self.apicalTrunk_ext = []
        apicalTrunk_ext = apicalTrunk+[f'apic[{i}]' for i in [36,40,53,61]]

        self.apicalTuft = []
        apicalTuft = [f'apic[{i}]' for i in [37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,54,55,56,57,58,59,60,61,62,63,64,65,66,67]]

        self.apical_obliques = []
        obliques = [f'apic[{i}]' for i in [1,2,3,4,5,6,7,10,12,14,15,16,17,18,20,22,24,25,26,28,29,30,32,33,34,68,69,70,71,72,73,74,75,76,77,78]]

        for x in self.all:
            self.allsec.append(x)
            if (not 'soma' in str(x)) and (not 'axon' in str(x)):
                self.alldend.append(x)
            if any([y in str(x) for y in apicalTrunk]):
                self.apicalTrunk.append(x)
            if any([y in str(x) for y in apicalTrunk_ext]):
                self.apicalTrunk_ext.append(x)
            if any([y in str(x) for y in apicalTuft]):
                self.apicalTuft.append(x)
            if any([y in str(x) for y in obliques]):
                self.apical_obliques.append(x)

class INT_cAC_noljp4(NeuronTemplate):
    def __init__ (self,**kwargs):
        super().__init__(templatepath = './Model/cell_seed4_0-cac-06.hoc', templatename = 'INT_cAC_noljp4',**kwargs)
        self.load_template()
        self.move_attributes()
        self.make_lists()
        self.moveSomaToOrigin()
        self.rotate_Cell(init_rotation=True)
    def __str__ (self):
        try:
            return f'compartCell_{self.__class__.__name__}_{self.ID}'
        except: return 'compartCell%d'%self.ID
    def __repr__ (self):
        return self.__str__()
    def make_lists(self):
        self.allsec = []
        self.alldend = []
        self.apicalTrunk = []
        apicalTrunk = [f'apic[{i}]' for i in [0,8,9,11,13,19,21,23,27,31,35]]

        self.apicalTrunk_ext = []
        apicalTrunk_ext = apicalTrunk+[f'apic[{i}]' for i in [36,40,53,61]]

        self.apicalTuft = []
        apicalTuft = [f'apic[{i}]' for i in [37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,54,55,56,57,58,59,60,61,62,63,64,65,66,67]]

        self.apical_obliques = []
        obliques = [f'apic[{i}]' for i in [1,2,3,4,5,6,7,10,12,14,15,16,17,18,20,22,24,25,26,28,29,30,32,33,34,68,69,70,71,72,73,74,75,76,77,78]]

        for x in self.all:
            self.allsec.append(x)
            if (not 'soma' in str(x)) and (not 'axon' in str(x)):
                self.alldend.append(x)
            if any([y in str(x) for y in apicalTrunk]):
                self.apicalTrunk.append(x)
            if any([y in str(x) for y in apicalTrunk_ext]):
                self.apicalTrunk_ext.append(x)
            if any([y in str(x) for y in apicalTuft]):
                self.apicalTuft.append(x)
            if any([y in str(x) for y in obliques]):
                self.apical_obliques.append(x)

class INT_cAC_noljp3(NeuronTemplate):
    def __init__ (self,**kwargs):
        super().__init__(templatepath = './Model/cell_seed7_0-cac-04.hoc', templatename = 'INT_cAC_noljp3',**kwargs)
        self.load_template()
        self.move_attributes()
        self.make_lists()
        self.moveSomaToOrigin()
        self.rotate_Cell(init_rotation=True)
    def __str__ (self):
        try:
            return f'compartCell_{self.__class__.__name__}_{self.ID}'
        except: return 'compartCell%d'%self.ID
    def __repr__ (self):
        return self.__str__()
    def make_lists(self):
        self.allsec = []
        self.alldend = []
        self.apicalTrunk = []
        apicalTrunk = [f'apic[{i}]' for i in [0,8,9,11,13,19,21,23,27,31,35]]

        self.apicalTrunk_ext = []
        apicalTrunk_ext = apicalTrunk+[f'apic[{i}]' for i in [36,40,53,61]]

        self.apicalTuft = []
        apicalTuft = [f'apic[{i}]' for i in [37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,54,55,56,57,58,59,60,61,62,63,64,65,66,67]]

        self.apical_obliques = []
        obliques = [f'apic[{i}]' for i in [1,2,3,4,5,6,7,10,12,14,15,16,17,18,20,22,24,25,26,28,29,30,32,33,34,68,69,70,71,72,73,74,75,76,77,78]]

        for x in self.all:
            self.allsec.append(x)
            if (not 'soma' in str(x)) and (not 'axon' in str(x)):
                self.alldend.append(x)
            if any([y in str(x) for y in apicalTrunk]):
                self.apicalTrunk.append(x)
            if any([y in str(x) for y in apicalTrunk_ext]):
                self.apicalTrunk_ext.append(x)
            if any([y in str(x) for y in apicalTuft]):
                self.apicalTuft.append(x)
            if any([y in str(x) for y in obliques]):
                self.apical_obliques.append(x)


def _colorsecs(mylist,seclist,cell):
    # function used in debug mode to color sections -> identify which sections part of seclist
    #mylist =  ['apic[0]','apic[6]','apic[8]','apic[10]','apic[12]','apic[18]','apic[20]','apic[24]','apic[26]','apic[30]','apic[46]','apic[60]','apic[62]','apic[66]','apic[68]','apic[70]'] + ['apic[31]','apic[33]','apic[39]','apic[47]','apic[51]']
    #_colorsecs(mylist,'apical_obliques',cell)
    plt.close()
    setattr(cell,seclist,[])
    mylist = [f'apic[{i}]' for i in mylist]
    for x in cell.allsec:
        if any([y in str(x) for y in mylist]):
            getattr(cell,seclist).append(x)
    
    cell.tf = []
    for sec in cell.allsec:
        for seg in sec:
            if sec in getattr(cell,seclist):
                cell.tf.append(1)
            else:
                cell.tf.append(0)
    fig = plt.figure()
    ax = plt.subplot(111,projection='3d')
    mphv2.shapeplot(h,ax,sections = cell.allsec,cvals =cell.tf,cmap='RdYlBu')
    ax.set_title(cell)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=90, azim=-90)
    ax.grid(visible=None)
    ax.axis('off')
    plt.show(block=False)

if __name__ == '__main__':
    if os.path.exists("./Model/Mods/x86_64/libnrnmech.so"):
        # linux compiled file (on hpc)
        h.nrn_load_dll("./Model/Mods/x86_64/libnrnmech.so")
        print("succes load libnrnmech.so")
    else:
        # above file should not exist locally -> load windows compiled nrnmech
        h.nrn_load_dll("./Model/Mods/nrnmech.dll")
        print("succes load nrnmech.dll")
    import Functions.globalFunctions.ExtracellularField as eF
    mpluse('tkagg')

    #stim settings
    delay=100; dur=100-1e-6; amp=3000; prf=10/1000; dc=0.1
    # sim settings
    simdur = 300 #ms
    dt = 0.025 #ms

    hShape_flag = False
    cell = locals()[NeuronTemplates[4]](replace_axon = True)
    cell.insertOptogenetics(cell.alldend)

    if not hShape_flag:
        h.Shape(False)
        hShape_flag = True

    #Gather sec positions before movement
    secpos = cell.gather_secpos()
    cell.move_Cell(-secpos.loc[['soma[0]']].to_numpy()[0]) # by default included
    cell.rotate_Cell(theta = np.pi/180*30)

    # gather new sec positions
    secpos2 = cell.gather_secpos()
    #combine in single dataframe
    for k in ['x','y','z']:
        secpos[k+'2'] = secpos2[k]
        secpos['d'+k] = secpos[k+'2']-secpos[k]
    #print
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(secpos)

    #Plot
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(111,projection='3d')
    ax = cell.sec_plot(ax)
    ax.set_title(cell.celltype)
    ax.set_zlim([-300,300])
    ax.set_xlim([-300,300])
    ax.set_ylim([-200,400])
    ax.view_init(elev=90, azim=-90)
    plt.show(block = False)

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
    structured = True
    # define norm => all plots have same colorscale
    norm = cm.colors.Normalize(vmax=abs(field[:,3]).max(), vmin=field[:,3].min())
    lognorm = cm.colors.LogNorm(vmax=abs(field[:,3]).max(), vmin=field[field[:,3]>0,3].min())


    # init figure and subplots
    fig,axs = plt.subplots(5,5,figsize = (16,9))

    poss = np.append(np.linspace(-50,-2,int((axs.size-2)/2),endpoint=False).round(2),np.linspace(0,50,int(np.ceil((axs.size-2)/2)),endpoint=True).round(2))
    poss = np.append(poss,[-2,-1])
    poss = np.sort(poss)

    gridorder = eF.checkGridOrder(field[:,0],field[:,1])
    field = eF.prepareDataforInterp(field,'ninterp')
    xX,xY,xZ = np.meshgrid(field[0],field[1],field[2],indexing=gridorder)
    eF.slicePlot(field,axs,fig,poss,(xX,xY,xZ),norm=norm,showfig=False,xyz='xy',structured = True)
    fig,axs = plt.subplots(5,5,figsize = (16,9))
    eF.slicePlot(field,axs,fig,poss,(xX,xY,xZ),norm=norm,showfig=False,xyz='yz',structured = True)
    fig,axs = plt.subplots(5,5,figsize = (16,9))
    eF.slicePlot(field,axs,fig,poss,(xX,xY,xZ),norm=lognorm,showfig=False,xyz='xy',structured = True)

    # apply field to Neuron
    cell.updateXtraCoors()
    attach_flag,totalos = eF.attach_stim(cell.allsec, field, structured, netpyne=False, stimtype='optical', phi = np.pi/2,xT = [0,0,-secpos['y2'].min()])

    #update chr2 gbar distribution
    from scipy.stats import truncnorm
    scale = 200
    a, b = (-1000 - 0)/scale, (1000 - 0)/scale
    source = [300,300,300]
    rv = truncnorm(a,b,0,scale)
    #distribution = lambda seg_xyz: 5*rv.pdf(np.linalg.norm(np.array(seg_xyz)-source))/rv.pdf(0)
    distribution = lambda x: 5
    seglist,valueschr2 = cell.distribute_mechvalue(distribution)
    swv,swov = cell.updateMechValue(seglist,valueschr2,mech = 'gchr2bar_chr2h134r')
    print(swov)

    fig = plt.figure(figsize = (16,10))
    ax = plt.subplot(131,projection='3d')
    mphv2.shapeplot(h,ax,cvals_type='os',colorscale='log10')
    ax.set_title(cell)
    ax.set_zlim([-300,300])
    ax.set_xlim([-300,300])
    ax.set_ylim([-200,400])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=90, azim=-90)
    #ax.grid(visible=None)
    #ax.axis('off')

    ax = plt.subplot(132,projection='3d')
    mphv2.shapeplot(h,ax,cvals_type='gchr2bar_chr2h134r')
    ax.set_title(cell)
    ax.set_zlim([-300,300])
    ax.set_xlim([-300,300])
    ax.set_ylim([-200,400])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=90, azim=-90)
    ax.grid(visible=None)
    ax.axis('off')

    # test on how to create extra colorcoded figure
    connpoints = []
    parentsegs = []
    diams = []
    cpdiams = []
    dfdiams2 =[]
    nrs = []
    for sec in cell.alldend:
        parentseg = sec.parentseg()
        parentsegs.append(parentseg)
        cpdiams.append((sec(0).diam,parentseg.diam))
        for i in range(sec.n3d()):
            diams.append((str(sec(i/(sec.n3d()-1))),sec.diam3d(i)))
        for seg in sec:
            connpoints.append(parentseg.x)
            dfdiams2.append(-1*(sec(0).diam-parentseg.diam))
            nrs.append(int(str(sec).rsplit('[',1)[-1][:-1]))
            if 'soma' in str(parentseg):
                dfdiams2[-1] = 0
    ax = plt.subplot(133,projection='3d')
    mphv2.shapeplot(h,ax,sections = cell.alldend,cvals = dfdiams2)
    ax.set_title(cell)
    ax.set_zlim([-300,300])
    ax.set_xlim([-300,300])
    ax.set_ylim([-200,400])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=90, azim=-90)
    ax.grid(visible=None)
    ax.axis('off')


    # setup stimulation
    if attach_flag:
        t_pt = np.arange(0,simdur*1.1, dt/10)
        stim_time, stim_amp = eF.pulseTrain(t_pt,delay = delay, dur=dur, prf=prf, dc = dc, amp=amp)

        if stim_amp is not None:
            stim_amp = h.Vector(stim_amp)
            stim_time = h.Vector(stim_time)
            stim_amp.play(h._ref_ostim_xtra, stim_time, True) #True

    # setup recording
    Dt = 0.1
    allv_traces = []
    names = []
    t = h.Vector().record(h._ref_t,Dt)
    v_s = h.Vector().record(cell.dend[0](0.5)._ref_v,Dt)
    i_chr2 = h.Vector().record(cell.dend[0](0.5)._ref_i_chr2h134r,Dt)
    for x in cell.allsec:
        allv_traces.append(h.Vector().record(x(0.5)._ref_v,Dt))
        names.append(str(x))

    #do simulation
    h.dt = dt
    cell.check_pointers(True)
    h.finitialize(-60)
    h.continuerun(simdur)

    # plot traces
    fig,axs = plt.subplots(2,1,sharex=True,tight_layout=True,figsize=(9,6))
    axs[0].plot(t,v_s)
    axs[1].plot(t,i_chr2)

    # fill locations of pulses
    Iopt_np = np.array(stim_amp)
    idx_on = (Iopt_np>0) & (np.roll(Iopt_np,1)<=0)
    idx_off = (Iopt_np>0) & (np.roll(Iopt_np,-1)<=0)
    t_np = np.array(stim_time)
    t_on = t_np[idx_on]
    t_off = t_np[idx_off]
    if len(t_on)>len(t_off):
        # if illumination till end of simulaton time t_off could miss a final time point
        t_off = np.append(t_off,t_np[-1])
    for ton,toff in zip(t_on,t_off):
        axs[1].axvspan(ton,toff,color='tab:blue',alpha=0.2)

    #set labels
    axs[1].set_xlim([0,simdur])
    axs[1].set_xlabel('time [ms]')
    axs[1].set_ylabel('ichr2 [uA/cm2]')
    axs[0].set_ylabel('V [mV]')

    # plot all recorded v traces
    fig = plt.figure()
    nrfigs = int(np.ceil(len(names)/4))
    axs = fig.subplots(nrfigs,1)
    if not isinstance(axs,np.ndarray):
        axs = [axs]
    for i,ax in enumerate(axs):
        t1 = i*4
        t2 = (i+1)*4
        for vtrace, name in zip(allv_traces[t1:t2],names[t1:t2]):
            ax.plot(t,vtrace,label=name.split('.')[-1])
        ax.legend()
    
    plt.show()
