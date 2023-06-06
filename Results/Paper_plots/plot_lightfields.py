import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from neuron import h

import Functions.globalFunctions.ExtracellularField as EcF
import Functions.globalFunctions.morphology_v2 as mphv2
from Model import Cells

cmap_contour = sns.color_palette("mako", as_cmap=True)

figw = 3.7*3

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

lightfields = glob.glob(
    './Inputs/LightIntensityProfile/Ugent470grayinvivo_EET/*.txt')
print(lightfields)
sim_idx = [int(x.rsplit('_', 1)[-1].split('.', 1)[0]) for x in lightfields]
muas = [float(x.rsplit('_mua', 1)[-1].split('_', 1)[0]) for x in lightfields]
redmus = [float(x.rsplit('_redmus', 1)[-1].split('_', 1)[0])
          for x in lightfields]

print(sim_idx)
print(lightfields[sim_idx.index(0)])
print(lightfields[sim_idx.index(1)])
print(lightfields[sim_idx.index(2)])
print(muas[sim_idx.index(0)])
print(redmus[sim_idx.index(0)])
print(lightfields[muas.index(max(muas))])
print(lightfields[muas.index(min(muas))])
print(lightfields[redmus.index(max(redmus))])
print(lightfields[redmus.index(min(redmus))])
idx_default = sim_idx.index(0)
idx_maxmua = muas.index(max(muas))
idx_maxmus = redmus.index(max(redmus))
xlims = [-0.75, 0.75]
ylims = [-0.2, 1.3]
myfields = []
for x in [idx_default, idx_maxmua, idx_maxmus]:
    myfield = np.genfromtxt(
        lightfields[x], comments='%')
    myfield = myfield.copy()
    myfield[:, :-1] = myfield[:, :2]/1000
    myfield[myfield[:, -1] < 1e-7, -1] = 1e-7
    idx = (myfield[:, 0] >= xlims[0]) & (myfield[:, 0] <= xlims[1]) & (
        myfield[:, 1] >= ylims[0]) & (myfield[:, 1] <= ylims[1])
    myfield = myfield[idx, :]
    myfields.append(myfield)
levels = [1e-3, 1e-2, 1e-1, 0.5]
norm = mpl.colors.LogNorm(vmax=np.max(
    [np.max(x[:, -1]) for x in myfields]), vmin=8e-4)

fig, axs = plt.subplots(1, len(myfields), figsize=(
    14/2.54, 14/3.5/2.54), sharey=True)
for myfield, ax in zip(myfields, axs):
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
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    im.set_linewidth(0)
    im.set_linewidths(0)
    im.set_rasterized(True)
    im.set_edgecolor('face')

    ax.legend(handle[::-1], [f'{np.log10(x)*10:0.0f} dB' for x in levels][::-1],
              ncol=3, loc='upper right')
    ax.invert_yaxis()
for ax in axs[:-1]:
    ax.get_legend().remove()
for ax in axs:
    ax.set_box_aspect(1)
# fig.tight_layout()
fig.savefig('lightfields.svg', dpi=300)
plt.show()
