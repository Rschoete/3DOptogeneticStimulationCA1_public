import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from neuron import h

import Functions.globalFunctions.morphology_v2 as mphv2
from Model import Cells

font = {'family': 'helvetica',
        'size': 10}

mpl.rc('font', **font)

image_w = 9

pttocm = 0.0352777778
scale_diams = image_w/pttocm/1100*5
print(scale_diams)

colorlist = []


def assign_colors(allsec, colorkeyval):
    colorlist = []
    for sec in allsec:
        for seg in sec:
            if 'soma' in str(sec):
                colorlist.append(colorkeyval['soma'])
            elif 'axon' in str(sec):
                colorlist.append(colorkeyval['axon'])
            elif 'apic' in str(sec):
                colorlist.append(colorkeyval['apic'])
            elif 'dend' in str(sec):
                colorlist.append(colorkeyval['dend'])
            else:
                colorlist.append([0, 0, 0])
    return colorlist


def set_size(w, h, ax=None):
    """ w, h: width, height in inches """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw/2.54, figh/2.54)


h.nrn_load_dll("./Model/Mods/nrnmech.dll")
print("succes load nrnmech.dll")
pyr1_str = Cells.NeuronTemplates[0]
pyr2_str = Cells.NeuronTemplates[1]
int1_str = Cells.NeuronTemplates[4]
int2_str = Cells.NeuronTemplates[5]


pyr1 = getattr(Cells, pyr1_str)(replace_axon=False,
                                movesomatoorigin=False, allign_axis=False)
pyr2 = getattr(Cells, pyr2_str)(replace_axon=False,
                                movesomatoorigin=False, allign_axis=False)
int1 = getattr(Cells, int1_str)(replace_axon=False,
                                movesomatoorigin=False, allign_axis=False)
int2 = getattr(Cells, int2_str)(replace_axon=False,
                                movesomatoorigin=False, allign_axis=False)

cells = [pyr1, pyr2, int1, int2]  # , int2]
for x in cells:
    x.movesomatoorigin = True
    x.moveSomaToOrigin()
    x.allign_cell_toaxis()
# pyr2.moveSomaToOrigin()
# int1.moveSomaToOrigin()
# int2.moveSomaToOrigin()

#cell2 = getattr(Cells, NeuronTemplates[1])(replace_axon=False)
fig = plt.figure(figsize=(7, 7))
ax = plt.subplot(111, projection='3d')
def myfun(x, y): return list(mpl.colors.to_rgb(x))+[y]


colorkeyval1 = {'soma': myfun('tab:red', 1), 'axon': myfun('tomato', 1), 'apic': myfun('seagreen', 0.6),
                'dend': myfun('teal', 0.6), 'unclassified': [0, 0, 0]}
colorkeyval2 = {'soma': myfun('tab:red', 1), 'axon': myfun('tomato', 0.2),
                'dend': myfun('skyblue', 0.5), 'unclassified': [0, 0, 0]}

colorkeyval = colorkeyval2
for i, (x, movey) in enumerate(zip(cells[::-1], [0, 200, -200, 400][::-1])):
    print(f'\t* celltype: {x.celltype}\n\t* morphology: {x.morphology}')
    x.rotate_Cell(theta=-np.pi/2)
    x.move_Cell([0, movey, 0])
    if i >= 2:
        colorkeyval = colorkeyval1
    scale_diams_list = [scale_diams]*len(x.allsec)
    scale_diams_list[-1] = scale_diams_list[-1]/2

    colorlist = assign_colors(x.allsec[::-1], colorkeyval=colorkeyval)
    mphv2.shapeplot(h, ax, sections=x.allsec[::-1],
                    cvals=colorlist, cb_flag=False, clim=[0, 0], scale_diams=scale_diams_list)


ax.set_zlim([-400, 700])
ax.set_xlim([-500, 500])
ax.set_ylim([-550, 550])
ax.invert_zaxis()
ax.view_init(elev=0, azim=0)
# ax.get_legend().remove()
ax.set_zlim([-400, 700])
ax.set_xlim([-500, 500])
ax.set_ylim([-550, 550])
ax.invert_zaxis()
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.plot([-400, -400], [-400, -300], zs=0,
#         zdir='x', color=[0.2, 0.2, 0.2], alpha=0.7)
# ax.plot([-400, -300], [-400, -400], zs=0,
#         zdir='x', color=[0.2, 0.2, 0.2], alpha=0.7)
ax.view_init(elev=0, azim=0)
set_size(image_w, image_w, ax=ax)
ax.set_zlabel('z', rotation=0)

legendkv = colorkeyval2
del legendkv['unclassified']
legendkv['basal'] = colorkeyval1['dend']
legendkv['apical'] = colorkeyval1['apic']
for k, v in legendkv.items():
    ax.plot(np.nan, np.nan, np.nan, label=k, color=v)
    ax.legend(frameon=False, loc='upper left',
              bbox_to_anchor=(0.15, 0.5), handlelength=1)
plt.show()
fig.savefig('ca1cells.svg')
