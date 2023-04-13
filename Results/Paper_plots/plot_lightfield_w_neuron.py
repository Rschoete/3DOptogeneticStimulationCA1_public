import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from neuron import h

import Functions.globalFunctions.ExtracellularField as EcF
import Functions.globalFunctions.morphology_v2 as mphv2
from Model import Cells

cmap_contour = sns.color_palette("mako", as_cmap=True)
cmap_neuron = sns.light_palette("teal", as_cmap=True)

figw = 3.7

font = {'family': 'helvetica',
        'size': 10}


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


mpl.rc('font', **font)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

myfield_init = np.genfromtxt(
    './Inputs/LightIntensityProfile/Ugent470nIrr_np1e7_res5emin3_gf1_cyl_5x10.txt', comments='%')
myfield = myfield_init.copy()
myfield[:, :-1] = myfield[:, :2]/1000
myfield[myfield[:, -1] < 1e-7, -1] = 1e-7
levels = [1e-4, 1e-3, 1e-2, 1e-1, 0.5]
norm = mpl.colors.LogNorm(vmax=np.max(
    myfield[:, -1]), vmin=1e-6)
r = np.linspace(-np.max(myfield[:, 0]), np.max(myfield[:, 0]), 100)
z = np.linspace(np.min(myfield[:, 1]), np.max(myfield[:, 1]), 100)
rR, zZ = np.meshgrid(r, z)
myfield = EcF.prepareDataforInterp(myfield, 'ninterp')
# add mirror around zaxis
# output myfield is 'ij'
myfield2 = (np.concatenate((-np.flip(myfield[0][1:]), myfield[0])),
            myfield[1],
            np.vstack((np.flipud(myfield[-1][1:, :]), myfield[-1]))
            )
im, cntr, cb = EcF.slicePlot(myfield2, ax, fig, plotGrid=(
    rR, zZ), structured=True, norm=norm, contour_kwargs={'norm': norm, 'levels': levels, 'cmap': cmap_contour}, cmap='Blues')
handle, l = cntr.legend_elements("I")
cb.remove()
im.set_linewidth(0)
im.set_linewidths(0)
im.set_rasterized(True)
im.set_edgecolor('face')

ax.legend(handle[::-1], [f'{np.log10(x)*10:0.0f} dB' for x in levels][::-1],
          ncol=3, loc='upper right')
ax.invert_yaxis()
ax.set_yticks([-4, -2, 0, 2, 4, 6])
ax.set_box_aspect(1)
set_size(figw, figw, ax=ax)
fig.savefig('OpticField.svg', dpi=300)

h.nrn_load_dll("./Model/Mods/nrnmech.dll")
print("succes load nrnmech.dll")


fig = plt.figure(figsize=(7, 7))
ax = plt.subplot(111, projection='3d')

pyr1_str = Cells.NeuronTemplates[0]
pyr1 = getattr(Cells, pyr1_str)(replace_axon=False,
                                movesomatoorigin=True, allign_axis=True)
pyr1.rotate_Cell(theta=-np.pi/2)
pyr1.move_Cell([0, 0, 400])

pyr1.insertOptogenetics(seclist=pyr1.allsec,
                        opsinmech='chr2h134r', set_pointer_xtra=True)
pyr1.updateXtraCoors()
pyr1.check_pointers(True)
t = np.arange(0, 100*1.1, 0.01/10)
myfield = EcF.prepareDataforInterp(myfield_init, 'ninterp')
EcF.setXstim(pyr1.allsec, t, 10, 10, 1, myfield, True,
             'singleSquarePulse', stimtype='optical', netpyne=False)
pttocm = 0.0352777778
scale_diams = 4/pttocm/1100*5
scale_diams_list = [scale_diams]*len(pyr1.allsec)
scale_diams_list[-1] = scale_diams_list[-1]
_, cbar = mphv2.shapeplot(h, ax, pyr1.allsec[-1::-1], cvals_type='os_xtra',
                          cmap=cmap_neuron, colorscale='log10', scale_diams=scale_diams_list)
ax.grid(False)
cbar.ax.set_yticks([-0.5, -1, -1.5, -2])
cbar.ax.set_yticklabels([-5, -10, -15, -20])

ax.set_xticks([])
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_zlim([200, 850])
ax.set_xlim([-200, 450])
ax.set_ylim([-200, 450])
ax.view_init(elev=0, azim=0)
set_size(figw*1.5, figw*1.5, ax=ax)
ax.set_zticklabels([f'{x/1000:0.1f}' for x in ax.get_zticks()])
ax.set_yticklabels([f'{x/1000:0.1f}' for x in ax.get_yticks()])
ax.invert_zaxis()
plt.show()
fig.savefig('neuroninfield.svg', dpi=300)
