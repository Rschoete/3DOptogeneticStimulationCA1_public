import glob
import json
import os
import sys
import time
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy import integrate

import Functions.globalFunctions.ExtracellularField as EcF
from Analyses.SDC.tools import VTA2D

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CODE IS NOT GENERIC DESIGNED TO COLLECT DATA FROM FOLDER WHERE SCRIPT IS LOCATED
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# opsinlocations to single word
opsinLocation_map = {'pc': {'1000': 'soma', '0100': 'axon', '0010': 'apic',
                     '0001': 'dend', '0011': 'alldend', '1111': 'allsec'},
                     'int': {'100': 'soma', '010': 'axon',
                             '001': 'dend', '111': 'allsec'}}


def load_data_df(*, filepath, filename, recollect, result, all_columns, cell_init_options, settings_options, opsin_options, field_options, fill_missing_xyzpositions=True, save_recollect=True, savename=None):
    if recollect:
        master_df = _recollect_data_df(filepath=filepath, fill_missing_xyzpositions=fill_missing_xyzpositions, result=result, all_columns=all_columns, cell_init_options=cell_init_options,
                                       settings_options=settings_options, opsin_options=opsin_options, field_options=field_options, savename=savename, save_recollect=save_recollect)
    else:
        master_df = pd.read_csv(os.path.join(filepath, filename), index_col=0)
    return master_df


def _recollect_data_df(*, filepath, fill_missing_xyzpositions, result, all_columns, cell_init_options, settings_options, opsin_options, field_options, savename, save_recollect):
    if savename is None:
        savename = os.path.join(filepath, 'all_data.csv')
    if os.path.exists(savename):
        savename = '(2).'.join(savename.rsplit('.', 1))

    # list all data and input file locations
    alldata_files = [x for x in glob.glob(os.path.join(
        filepath, '**/data.json'), recursive=True) if not 'ToConcat' in x]
    allinput_files = [x for x in glob.glob(os.path.join(
        filepath, '**/input.json'), recursive=True) if not 'ToConcat' in x]
    dir_list = [x for x in glob.glob(os.path.join(
        filepath, '*')) if os.path.isdir(x) and not 'ToConcat' in x]
    print(
        f"result files: {len(alldata_files)}, input files: {len(allinput_files)}, directories: {len(dir_list)}")

    # init master_df: dataframe where all results will be stored
    master_df = pd.DataFrame(columns=all_columns)

    for data_path, input_path in zip(alldata_files, allinput_files):
        # print(data_path,input_path,)
        # Load data and input file
        with open(data_path, 'r') as f:
            mydata = json.load(f)
        with open(input_path, 'r') as f:
            myinput = json.load(f)

        # store global input info
        cell_init_dict = {'neurontemplate': myinput['settings']['cellsopt']['neurontemplate'], **{key: myinput['settings']['cellsopt']['init_options'][key.rsplit(
            '_', 1)[0]] for key in cell_init_options if key != 'neurontemplate'}}
        settings_dict = {
            key: myinput['settings'][key] for key in settings_options}
        opsin_dict = {**{opsin_options[0]: myinput['settings']['cellsopt']['opsin_options'][opsin_options[0]+'_total']}, **{
            key: myinput['settings']['cellsopt']['opsin_options'][key] for key in opsin_options[1:]}}
        field_dict = {field_options[0]: myinput['settings']['stimopt']['Ostimparams']['filepath'].rsplit('/', 1)[-1].rsplit('.txt', 1)[0],
                      field_options[1]: myinput['settings']['analysesopt']['SDOptogenx']['r_p'.join(field_options[1].split('P'))+'OI']}

        if 'pc' in cell_init_dict['neurontemplate'].lower():
            opsin_dict['opsinlocations'] = opsinLocation_map['pc'][''.join(
                [str(int(any([loc in oloc for oloc in opsin_dict['opsinlocations']]))) for loc in list(opsinLocation_map['pc'].values())[:4]])]
            opsin_dict['opsinlocations'] = 'basaldend' if opsin_dict['opsinlocations'] == 'dend' else opsin_dict['opsinlocations']
        else:
            opsin_dict['opsinlocations'] = opsinLocation_map['int'][''.join(
                [str(int(any([loc in oloc for oloc in opsin_dict['opsinlocations']]))) for loc in list(opsinLocation_map['int'].values())[:3]])]
            opsin_dict['opsinlocations'] = 'alldend' if opsin_dict['opsinlocations'] == 'dend' else opsin_dict['opsinlocations']
        info_dict = {**field_dict, **opsin_dict,
                     **cell_init_dict, **settings_dict}

        # store simulation specific info and add to dataframe
        durs = myinput['settings']['analysesopt']['SDOptogenx']["durs"]

        for amp, sR, ichr2, dur in zip(*mydata["SDcurve"]["Optogenx"], durs):
            result_dict = {result[0]: amp, result[1]: sR, result[2]: ichr2['abs_1']
                           ['total'], result[3]: ichr2['abs_1']['total_g'], result[-1]: dur}
            intm = pd.DataFrame(
                {**result_dict, **info_dict}, index=[0], columns=all_columns)
            master_df = pd.concat([master_df, pd.DataFrame(
                {**result_dict, **info_dict}, index=[0], columns=all_columns)])
        master_df = master_df.reset_index(drop=True)

    if save_recollect:
        master_df.to_csv(savename)
    return master_df


if __name__ == '__main__':
    recollect = True
    fill_missing_xyzpositions = False

    # Load data
    filepath = './Results/SDC/SDC_constI'
    filename = 'all_data.csv'

    # list parameters of interest
    cell_init_options = ['phi_0', 'theta_0', 'psi_0', 'neurontemplate']
    settings_options = ['seed', 'celsius', 'dt']
    opsin_options = ['Gmax', 'distribution', 'opsinmech',
                     'distribution_method', 'opsinlocations']
    field_options = ['field', 'nPulse']
    result = ['amp', 'sR', 'ichr2', 'gchr2', 'dur']
    all_columns = result+field_options+opsin_options + \
        cell_init_options+settings_options

    master_df = load_data_df(filepath=filepath, filename=filename, recollect=recollect, result=result, all_columns=all_columns, cell_init_options=cell_init_options,
                             settings_options=settings_options, opsin_options=opsin_options, field_options=field_options, fill_missing_xyzpositions=fill_missing_xyzpositions, save_recollect=True, savename=None)

    unique_values_columns = {
        key: master_df[key].unique() for key in all_columns}
    print(master_df.head())
