import Functions.globalFunctions.utils as utils
import Functions.globalFunctions.morphology_v2 as mphv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# https://matplotlib.org/stable/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
import matplotlib.colors as mcolors
import os
import glob  # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
from neuron import h

h.nrn_load_dll("./Model/Mods/nrnmech.dll")


# %%
def crop(image):
    mask = np.sum(image[:, :, 0:3], axis=2) == 3  # identify white pixels
    cc = np.all(mask, axis=0)  # columns of white pixels
    cc = (cc & ~np.roll(cc, 1)) | (cc & ~np.roll(cc, -1))
    cc = np.where(cc)  # get positions of boundary column white and non white
    if isinstance(cc, tuple):
        cc = cc[0]

    rc = np.all(mask, axis=1)  # rows of white pixels
    rc = (rc & ~np.roll(rc, 1)) | (rc & ~np.roll(rc, -1))
    rc = np.where(rc)
    if isinstance(rc, tuple):
        rc = rc[0]

    if len(cc) > 1:
        image = image[:, cc[0]+1:cc[-1], :]
    if len(rc) > 1:
        image = image[rc[0]+1:rc[-1], :, :]

    return image


# %%
cell_files = glob.glob("./Model/cell*.hoc")

iter = -1
for x in cell_files[:]:
    iter += 1
    x = x.replace('\\', '/')
    # find template name
    with open(x) as f:
        for line in f.readlines():
            # Find the start of the word
            index = line.find('endtemplate')
            # If the word is inside the line
            if index != -1:
                index += len('endtemplate ')
                attrname = line[index:]

    h.load_file(x)  # Load cell info

    cell = getattr(h, attrname)('./Model/morphologies')  # initial cell
    h.Shape(False)

    # print secpostitions
    secpos = {}
    for sec in cell.all:
        sec_name = str(sec).split('.')[-1]
        xyz = mphv2.get_section_path(h, sec)
        xyz = np.mean(xyz, axis=0)
        secpos[sec_name] = xyz
    secpos = pd.DataFrame(secpos).T.rename(
        lambda x: ['x', 'y', 'z'][x], axis=1)
    secpos = secpos.applymap(lambda x: utils.signif(x, 2))
    # more options can be specified also
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(secpos)

    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')
    mphv2.shapeplot(h, ax)
    ax.set_title(cell)
    ax.set_zlim([0, 100])
    ax.set_xlim([x+iter*500 for x in [-190, 200]])
    ax.set_ylim([-200, 200])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=90, azim=-90)

    # deleting previous cell sections doesn't work. do not know how to fix
    for sec in cell.all:
        sec = None
        del sec
    cell = None
    del cell
    for x in h.allsec():
        x = None
        del x
    for x in h.allsec():
        x = None
        del x
        print('')
        l = []
        del l
plt.show()
