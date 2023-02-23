import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neuron import h

import Functions.globalFunctions.morphology_v2 as mphv2
import Functions.support as sprt
from Model import Cells

h.nrn_load_dll("./Model/Mods/nrnmech.dll")
print("succes load nrnmech.dll")


def det_impedance(pos, sec, v0, freqs, dur, extend_impedance: bool):
    imps = []
    for f in freqs:
        h.finitialize(v0)
        h.continuerun(dur)
        imp = h.Impedance()
        imp.loc(pos, sec=sec)
        imp.compute(f, extend_impedance)
        ii = imp.input(pos, sec=sec)
        imps.append(float(ii))
    return imps


def _calc_imp(cell, sections=None, freqs=[0, 1, 10, 100], durinit=100, v0=-70, extend_impedance=False):
    # by default included
    all_imps = {}
    idx = -1
    if sections is None:
        sections = cell.allsec
    for sec in sections:
        print(f"{cell.allsec.index(sec)}/{len(sections)}", end='\r')
        for seg in sec:

            imps = det_impedance(
                seg.x, sec, v0, freqs, durinit, extend_impedance=extend_impedance)
            for imp, f in zip(imps, freqs):
                idx += 1
                all_imps[idx] = {'imp': imp, 'f': f, 'v0': v0,
                                 'durinit': durinit, 'neurontemplate': neurontemplate, 'seg': str(seg).split('.', 1)[-1]}

    imp_df = pd.DataFrame.from_dict(all_imps, orient='index')
    imp_df = imp_df.reset_index(drop=True)
    return imp_df


if __name__ == '__main__':
    calc_impedance_flag = True
    # impedance calculation before and after insertion of opsin in soma
    # has no effect. This is probably because Iopto is zero therfore gchr2 also 0
    include_opsin = False
    mpl.use('tkagg')
    plt.rcParams["font.family"] = "helvetica"

    Gmax_total = 1
    opsinlocations = 'soma'

    NeuronTemplates = ['CA1_PC_cAC_sig5',
                       'CA1_PC_cAC_sig6', 'cNACnoljp1', 'cNACnoljp2']

    neurontemplate = NeuronTemplates[2]
    print(f'Loading cell: {neurontemplate}')
    cell = getattr(Cells, neurontemplate)(replace_axon=False)
    #cell2 = getattr(Cells, NeuronTemplates[1])(replace_axon=False)
    print(f'\t* celltype: {cell.celltype}\n\t* morphology: {cell.morphology}')
    h.celsius = cell.celsius
    h.cvode.active(True)
    cell.rotate_Cell(theta=-np.pi/2)

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
    # imp_df = _calc_imp(cell, sections=cell.soma, freqs=[
    #                    0], extend_impedance=True)
    # print(imp_df)

    if include_opsin:
        # Insert Opsin
        opsinsections = sprt.convert_strtoseclist(cell, opsinlocations)
        cell.insertOptogenetics(seclist=opsinsections, set_pointer_xtra=True)
        G_total, seglist, values = cell.calc_Gmax_mechvalue(
            'gchr2bar_chr2h134r', values=None, seglist=[x for sec in opsinsections for x in sec])
        values = [val*Gmax_total/G_total for val in values]
        cell.updateMechValue(seglist, values, 'gchr2bar_chr2h134r')
        cell.updateXtraCoors()
        cell.check_pointers(True)
        imp_df = _calc_imp(cell, sections=cell.soma, freqs=[
                           0], extend_impedance=True)
        print(imp_df)
        # plot opsin distribution
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection='3d')
        mphv2.shapeplot(h, ax, cvals_type='gchr2bar_chr2h134r')
        ax.set_zlim([-400, 700])
        ax.set_xlim([-500, 500])
        ax.set_ylim([-550, 550])
        ax.invert_zaxis()
        ax.view_init(elev=0, azim=0)

    Cells._print_area(cell, ['allsec', 'soma', 'alldend', 'axon',
                             'apicalTrunk_ext', 'apicalTuft', 'basaldend', 'apical_obliques'])

    if calc_impedance_flag:
        imp_df = _calc_imp(cell, sections=cell.allsec, freqs=[
                           0, 1, 10, 100], extend_impedance=True)
        print(imp_df)
        imp_df.to_csv(f'impedance_{neurontemplate}.csv')
    plt.show()
