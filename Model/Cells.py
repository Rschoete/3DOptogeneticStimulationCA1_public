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
from matplotlib import use as mpluse

global hShape_flag

morphocelltype ={
    '990803': 'SP_PC','050921AM2':'SP_PC','mpg141017_a1-2_idC':'SP_PC', 'mpg141208_B_idA':'SP_PC', 'mpg141209_A_idA':'SP_PC', 'mpg141209_B_idA':'SP_PC','mpg141215_A_idA':'SP_PC','mpg141216_A_idA':'SP_PC','mpg141217_A_idB':'SP_PC','mpg150305_A_idB':'SP_PC','oh140521_B0_Rat_idA':'SP_PC','oh140521_B0_Rat_idC':'SP_PC','oh140807_A0_idA':'SP_PC','oh140807_A0_idB':'SP_PC','oh140807_A0_idC':'SP_PC','oh140807_A0_idF':'SP_PC','oh140807_A0_idG':'SP_PC','oh140807_A0_idH':'SP_PC','oh140807_A0_idJ':'SP_PC','010710HP2': 'SP_Ivy','011017HP2': 'SO_OLM','011023HP2': 'SO_BS','011127HP1': 'SLM_PPA','031031AM1': 'SP_CCKBC','060314AM2': 'SP_PVBC','970509HP2': 'SO_Tri','970627BHP1': 'SP_PVBC','970717D': 'SP_Ivy','970911C': 'SP_AA','971114B': 'SO_Tri','980120A': 'SO_BP','980513B': 'SP_BS','990111HP2': 'SP_PVBC','990611HP2': 'SR_SCA','990827IN5HP3': 'SR_IS1',
}
NeuronTemplates = ['CA1_PC_cAC_sig5','CA1_PC_cAC_sig6','bACnoljp8','bACnoljp7','cNACnoljp1','cNACnoljp2','INT_cAC_noljp4','INT_cAC_noljp3']

class NeuronTemplate:
    def __init__ (self,templatepath, templatename, replace_axon = True, morphologylocation = './Model/morphologies',ID=0,ty=0,col=0, phi = 0, theta = 0, insert_extracellular = False, set_pointer_xtra = False, movesomatoorigin = True):
        self.templatepath = templatepath
        self.templatename = templatename
        self.morphologylocation = morphologylocation
        self.replace_axon = str(replace_axon).lower()
        self.ID=ID
        self.ty=ty
        self.col=col
        self.phi = phi
        self.theta = theta
        self.insert_extracellular = insert_extracellular
        self.set_pointer_xtra = set_pointer_xtra
        self.movesomatoorigin = movesomatoorigin
    def load_template(self):
        self.assign_MorphandCelltype()
        h.load_file(self.templatepath) #Load cell info
        self.template = getattr(h,self.templatename)(self.replace_axon,self.morphologylocation) # initial cell
    def move_attributes(self):
        for x in self.template.__dir__():
            if x != '__class__':
                setattr(self,x,getattr(self.template,x))
        self.template = None
    def move_Cell(self,rt):
        for section in self.allsec:
            for i in range(section.n3d()):
                xyz = np.array([section.x3d(i),section.y3d(i),section.z3d(i)])
                xyz = xyz+rt
                h.pt3dchange(i, xyz[0], xyz[1], xyz[2], section.diam3d(i), sec=section)
    def rotate_Cell(self,phi=0,theta=0,init_rotation=False):

        if init_rotation:
            phi = self.phi
            theta = self.theta
        if phi!=0 or theta!=0:
            for section in self.allsec:
                for i in range(section.n3d()):
                    xyz = np.array([section.x3d(i),section.y3d(i),section.z3d(i)])
                    xyz = np.dot(np.dot(_Rz(phi),_Ry(theta)),xyz[:,None])
                    h.pt3dchange(i, xyz[0][0], xyz[1][0], xyz[2][0], section.diam3d(i), sec=section)

    def insertExtracellular(self):
        for sec in self.allsec:
            sec.insert('extracellular')
            if not sec.has_membrane('xtra'):
                sec.insert('xtra')
            if self.set_pointer_xtra:
                for seg in sec:
                    h.setpointer(seg.extracellular._ref_e, 'ex', seg.xtra)
    def insertOptogenetics(self,seclist,opsinmech = 'chr2h134r'):
        for sec in seclist:
            sec.insert(opsinmech)
            if not sec.has_membrane('xtra'):
                sec.insert('xtra')
            if self.set_pointer_xtra:
                for seg in sec:
                    h.setpointer(getattr(seg,opsinmech)._ref_Iopto, 'ox', seg.xtra)
    def updateXtraCoors(self):
        for sec in self.allsec:
            if sec.has_membrane('xtra'):
                pt3ds = [[getattr(sec,x)(i) for i in range(sec.n3d())] for x in ['x3d', 'y3d', 'z3d']]
                arcl = np.array([sec.arc3d(i) for i in range(sec.n3d())])
                arcl = arcl/arcl[-1] #normalize
                for seg in sec:
                    xyz_seg = [np.interp(seg.x,arcl,coors) for coors in pt3ds]
                    seg.x_xtra, seg.y_xtra, seg.z_xtra = xyz_seg
    def assign_MorphandCelltype(self):
        #https://stackoverflow.com/questions/31019854/typeerror-cant-use-a-string-pattern-on-a-bytes-like-object-in-re-findall
        with open(self.templatepath, 'r') as f:
            data = mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ).read().decode("utf-8")
            p = re.compile(r'load_morphology\("morphologies", .*\)')
            self.morphology = re.findall(r' (".*[(.asc)]")',p.search(data).group(0))[0][1:-5]
        self.celltype = morphocelltype[self.morphology]
    def moveSomaToOrigin(self):
        if self.movesomatoorigin:
            global hShape_flag
            if not hShape_flag:
                h.Shape(False)
                hShape_flag = True
            soma_pos = np.array([[self.soma[0].x3d(i),self.soma[0].y3d(i),self.soma[0].z3d(i)] for i in range(self.soma[0].n3d())])
            soma_pos = np.mean(soma_pos,axis=0)
            self.move_Cell(-soma_pos)

class CA1_PC_cAC_sig5(NeuronTemplate):
    def __init__ (self,**kwargs):
        super().__init__(templatepath = './Model/cell_seed4_0-pyr-04.hoc', templatename = 'CA1_PC_cAC_sig5',**kwargs)
        self.load_template()
        self.move_attributes()
        self.make_lists()
        self.moveSomaToOrigin()
        self.rotate_Cell(init_rotation=True)
        if self.insert_extracellular:
            self.insertExtracellular()
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
    def sec_plot(self,ax=None):
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(111,projection='3d')
        global hShape_flag
        if not hShape_flag:
            h.Shape(False)
            hShape_flag = True
        soma_pos = np.array([[self.soma[0].x3d(i),self.soma[0].y3d(i),self.soma[0].z3d(i)] for i in range(self.soma[0].n3d())])
        soma_pos = np.mean(soma_pos,axis=0)
        self.move_Cell(-soma_pos)

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
        mphv2.shapeplot(h,ax,sections = self.allsec,cvals =colorlist,cb_flag=False,clim=[0,0])
        ax.set_title('sec plot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.grid(visible=None)
        ax.axis('off')
        for k,v in colorkeyval.items():
            ax.plot(np.nan,np.nan,np.nan,label=k,color=v)
        ax.legend(frameon=False)
        return ax

class CA1_PC_cAC_sig6(NeuronTemplate):
    def __init__ (self, **kwargs):
        super().__init__(templatepath = './Model/cell_seed3_0-pyr-08.hoc', templatename = 'CA1_PC_cAC_sig6',**kwargs)
        self.load_template()
        self.move_attributes()
        self.make_lists()
        self.moveSomaToOrigin()
        self.rotate_Cell(init_rotation=True)
        if self.insert_extracellular:
            self.insertExtracellular()
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
    def sec_plot(self,ax=None):
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(111,projection='3d')
        global hShape_flag
        if not hShape_flag:
            h.Shape(False)
            hShape_flag = True
        soma_pos = np.array([[self.soma[0].x3d(i),self.soma[0].y3d(i),self.soma[0].z3d(i)] for i in range(self.soma[0].n3d())])
        soma_pos = np.mean(soma_pos,axis=0)
        self.move_Cell(-soma_pos)

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
        mphv2.shapeplot(h,ax,sections = self.allsec,cvals =colorlist,cb_flag=False,clim=[0,0])
        ax.set_title('sec plot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.grid(visible=None)
        ax.axis('off')
        for k,v in colorkeyval.items():
            ax.plot(np.nan,np.nan,np.nan,label=k,color=v)
        ax.legend(frameon=False)
        return ax

class bACnoljp8(NeuronTemplate):
    def __init__ (self,**kwargs):
        super().__init__(templatepath = './Model/cell_seed1_0-bac-10.hoc', templatename = 'bACnoljp8',**kwargs)
        self.load_template()
        self.move_attributes()
        self.make_lists()
        self.moveSomaToOrigin()
        self.rotate_Cell(init_rotation=True)

        if self.insert_extracellular:
            self.insertExtracellular()
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
    def sec_plot(self,ax=None):
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(111,projection='3d')
        global hShape_flag
        if not hShape_flag:
            h.Shape(False)
            hShape_flag = True
        soma_pos = np.array([[self.soma[0].x3d(i),self.soma[0].y3d(i),self.soma[0].z3d(i)] for i in range(self.soma[0].n3d())])
        soma_pos = np.mean(soma_pos,axis=0)
        self.move_Cell(-soma_pos)

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
        mphv2.shapeplot(h,ax,sections = self.allsec,cvals =colorlist,cb_flag=False,clim=[0,0])
        ax.set_title('sec plot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.grid(visible=None)
        ax.axis('off')
        for k,v in colorkeyval.items():
            ax.plot(np.nan,np.nan,np.nan,label=k,color=v)
        ax.legend(frameon=False)
        return ax

class cNACnoljp1(NeuronTemplate):
    def __init__ (self,**kwargs):
        super().__init__(templatepath = './Model/cell_seed1_0-cnac-04.hoc', templatename = 'cNACnoljp1',**kwargs)
        self.load_template()
        self.move_attributes()
        self.make_lists()
        self.moveSomaToOrigin()
        self.rotate_Cell(init_rotation=True)

        if self.insert_extracellular:
            self.insertExtracellular()
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
    def sec_plot(self,ax=None):
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(111,projection='3d')
        global hShape_flag
        if not hShape_flag:
            h.Shape(False)
            hShape_flag = True
        soma_pos = np.array([[self.soma[0].x3d(i),self.soma[0].y3d(i),self.soma[0].z3d(i)] for i in range(self.soma[0].n3d())])
        soma_pos = np.mean(soma_pos,axis=0)
        self.move_Cell(-soma_pos)

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
        mphv2.shapeplot(h,ax,sections = self.allsec,cvals =colorlist,cb_flag=False,clim=[0,0])
        ax.set_title('sec plot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.grid(visible=None)
        ax.axis('off')
        for k,v in colorkeyval.items():
            ax.plot(np.nan,np.nan,np.nan,label=k,color=v)
        ax.legend(frameon=False)
        return ax

class bACnoljp7(NeuronTemplate):
    def __init__ (self,**kwargs):
        super().__init__(templatepath = './Model/cell_seed2_0-bac-06.hoc', templatename = 'bACnoljp7',**kwargs)
        self.load_template()
        self.move_attributes()
        self.make_lists()
        self.moveSomaToOrigin()
        self.rotate_Cell(init_rotation=True)
        if self.insert_extracellular:
            self.insertExtracellular()
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
    def sec_plot(self,ax=None):
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(111,projection='3d')
        global hShape_flag
        if not hShape_flag:
            h.Shape(False)
            hShape_flag = True
        soma_pos = np.array([[self.soma[0].x3d(i),self.soma[0].y3d(i),self.soma[0].z3d(i)] for i in range(self.soma[0].n3d())])
        soma_pos = np.mean(soma_pos,axis=0)
        self.move_Cell(-soma_pos)

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
        mphv2.shapeplot(h,ax,sections = self.allsec,cvals =colorlist,cb_flag=False,clim=[0,0])
        ax.set_title('sec plot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.grid(visible=None)
        ax.axis('off')
        for k,v in colorkeyval.items():
            ax.plot(np.nan,np.nan,np.nan,label=k,color=v)
        ax.legend(frameon=False)
        return ax

class cNACnoljp2(NeuronTemplate):
    def __init__ (self,**kwargs):
        super().__init__(templatepath = './Model/cell_seed2_0-cnac-08.hoc', templatename = 'cNACnoljp2',**kwargs)
        self.load_template()
        self.move_attributes()
        self.make_lists()
        self.moveSomaToOrigin()
        self.rotate_Cell(init_rotation=True)
        
        if self.insert_extracellular:
            self.insertExtracellular()
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
    def sec_plot(self,ax=None):
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(111,projection='3d')
        global hShape_flag
        if not hShape_flag:
            h.Shape(False)
            hShape_flag = True
        soma_pos = np.array([[self.soma[0].x3d(i),self.soma[0].y3d(i),self.soma[0].z3d(i)] for i in range(self.soma[0].n3d())])
        soma_pos = np.mean(soma_pos,axis=0)
        self.move_Cell(-soma_pos)

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
        mphv2.shapeplot(h,ax,sections = self.allsec,cvals =colorlist,cb_flag=False,clim=[0,0])
        ax.set_title('sec plot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.grid(visible=None)
        ax.axis('off')
        for k,v in colorkeyval.items():
            ax.plot(np.nan,np.nan,np.nan,label=k,color=v)
        ax.legend(frameon=False)
        return ax

class INT_cAC_noljp4(NeuronTemplate):
    def __init__ (self,**kwargs):
        super().__init__(templatepath = './Model/cell_seed4_0-cac-06.hoc', templatename = 'INT_cAC_noljp4',**kwargs)
        self.load_template()
        self.move_attributes()
        self.make_lists()
        self.moveSomaToOrigin()
        self.rotate_Cell(init_rotation=True)
        if self.insert_extracellular:
            self.insertExtracellular()
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
    def sec_plot(self,ax=None):
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(111,projection='3d')
        global hShape_flag
        if not hShape_flag:
            h.Shape(False)
            hShape_flag = True
        soma_pos = np.array([[self.soma[0].x3d(i),self.soma[0].y3d(i),self.soma[0].z3d(i)] for i in range(self.soma[0].n3d())])
        soma_pos = np.mean(soma_pos,axis=0)
        self.move_Cell(-soma_pos)

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
        mphv2.shapeplot(h,ax,sections = self.allsec,cvals =colorlist,cb_flag=False,clim=[0,0])
        ax.set_title('sec plot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.grid(visible=None)
        ax.axis('off')
        for k,v in colorkeyval.items():
            ax.plot(np.nan,np.nan,np.nan,label=k,color=v)
        ax.legend(frameon=False)
        return ax

class INT_cAC_noljp3(NeuronTemplate):
    def __init__ (self,**kwargs):
        super().__init__(templatepath = './Model/cell_seed7_0-cac-04.hoc', templatename = 'INT_cAC_noljp3',**kwargs)
        self.load_template()
        self.move_attributes()
        self.make_lists()
        self.moveSomaToOrigin()
        self.rotate_Cell(init_rotation=True)
        if self.insert_extracellular:
            self.insertExtracellular()
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
    def sec_plot(self,ax=None):
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(111,projection='3d')
        global hShape_flag
        if not hShape_flag:
            h.Shape(False)
            hShape_flag = True
        soma_pos = np.array([[self.soma[0].x3d(i),self.soma[0].y3d(i),self.soma[0].z3d(i)] for i in range(self.soma[0].n3d())])
        soma_pos = np.mean(soma_pos,axis=0)
        self.move_Cell(-soma_pos)

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
        mphv2.shapeplot(h,ax,sections = self.allsec,cvals =colorlist,cb_flag=False,clim=[0,0])
        ax.set_title('sec plot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.grid(visible=None)
        ax.axis('off')
        for k,v in colorkeyval.items():
            ax.plot(np.nan,np.nan,np.nan,label=k,color=v)
        ax.legend(frameon=False)
        return ax


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

    import Functions.globalFunctions.morphology_v2 as mphv2
    import matplotlib.pyplot as plt
    import pandas as pd
    import Functions.globalFunctions.utils as utils
    from matplotlib import cm
    import Functions.globalFunctions.ExtracellularField as eF
    mpluse('tkagg')

    hShape_flag = False
    cell = locals()[NeuronTemplates[4]](replace_axon = True)
    cell.insertOptogenetics(cell.apical)
    if not hShape_flag:
        h.Shape(False)
        hShape_flag = True

    #Gather sec positions before movement
    secpos = {}
    for sec in cell.all:
        sec_name = str(sec).split('.')[-1]
        xyz = mphv2.get_section_path(h,sec)
        xyz = np.mean(xyz, axis=0)
        secpos[sec_name]=xyz
    secpos = pd.DataFrame(secpos).T.rename(lambda x:['x','y','z'][x], axis = 1)
    secpos = secpos.applymap(lambda x: utils.signif(x,2))
    cell.move_Cell(-secpos.loc[['soma[0]']].to_numpy()[0]) # by default included
    cell.rotate_Cell(theta = np.pi/180*30)

    # gather new sec positions
    secpos2 = {}
    for sec in cell.all:
        sec_name = str(sec).split('.')[-1]
        xyz = mphv2.get_section_path(h,sec)
        xyz = np.mean(xyz, axis=0)
        secpos2[sec_name]=xyz
    secpos2 = pd.DataFrame(secpos2).T.rename(lambda x:['x','y','z'][x], axis = 1)
    secpos2 = secpos2.applymap(lambda x: utils.signif(x,2))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(secpos2)
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
    plt.show()

