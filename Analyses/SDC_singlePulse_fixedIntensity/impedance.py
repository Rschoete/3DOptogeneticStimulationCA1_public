import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from neuron import h

import Functions.globalFunctions.morphology_v2 as mphv2
from Model import Cells

h.nrn_load_dll("./Model/Mods/nrnmech.dll")
print("succes load nrnmech.dll")


def det_impedance(pos, sec, v0, freqs, dur):
    imps = []
    for f in freqs:
        h.finitialize(v0)
        imp = h.Impedance()
        imp.loc(pos, sec=sec)
        h.continuerun(dur)
        imp.compute(f)
        ii = imp.input(pos, sec=sec)
        imps.append(float(ii))
    return imps


if __name__ == '__main__':
    mpl.use('tkagg')
    plt.rcParams["font.family"] = "helvetica"

    NeuronTemplates = ['CA1_PC_cAC_sig5',
                       'CA1_PC_cAC_sig6', 'cNACnoljp1', 'cNACnoljp2']

    neurontemplate = NeuronTemplates[0]
    print(f'Loading cell: {neurontemplate}')
    cell = getattr(Cells, neurontemplate)(replace_axon=False)
    #cell2 = getattr(Cells, NeuronTemplates[1])(replace_axon=False)
    print(f'\t* celltype: {cell.celltype}\n\t* morphology: {cell.morphology}')

    # cell.insertOptogenetics(cell.alldend)
    cell.rotate_Cell(theta=-np.pi/2)
    # cell2.rotate_Cell(theta=-np.pi/2)
    #cell2.move_Cell([0, 100, 0])

    # section Plot
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='3d')
    ax = cell.sec_plot(ax)
    #ax = cell2.sec_plot(ax)
    ax.set_title(cell.celltype)
    ax.set_zlim([-400, 700])
    ax.set_xlim([-500, 500])
    ax.set_ylim([-550, 550])
    ax.invert_zaxis()
    ax.view_init(elev=0, azim=0)

    Cells._print_area(cell, ['allsec', 'soma', 'alldend', 'axon',
                             'apicalTrunk_ext', 'apicalTuft', 'basaldend', 'apical_obliques'])
    # Cells._print_Gmax_info(cell)
    # Gather sec positions before movement
    secpos = cell.gather_secpos()
    print(max(secpos['z'])-min(secpos['z']))
    print(max(secpos['z'])/min(secpos['z']))
    # by default included
    all_imps = {}
    idx = -1
    freqs = [0, 1, 10, 100]
    durinit = 100
    v0 = -70
    for sec in cell.allsec:
        print(f"{cell.allsec.index(sec)}/{len(cell.allsec)}")
        for seg in sec:

            imps = det_impedance(
                seg.x, sec, v0, freqs, durinit)
            for imp, f in zip(imps, freqs):
                idx += 1
                all_imps[idx] = {'imp': imp, 'f': f, 'v0': v0,
                                 'durinit': durinit, 'neurontemplate': neurontemplate}
    import pandas as pd
    imp_df = pd.DataFrame.from_dict(all_imps, orient='index')
    imp_df = imp_df.reset_index(drop=True)
    imp_df.to_csv(f'impedance_{neurontemplate}.csv')
    plt.figure()
    plt.imshow(np.array(all_imps))
    plt.colorbar()

    plt.show()
