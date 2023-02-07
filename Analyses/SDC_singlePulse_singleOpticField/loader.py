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


def load_data_df(*, filepath, filename, recollect, result, all_columns, cell_init_options, settings_options, opsin_options, field_options, positional_input, fill_missing_xyzpositions=True, save_recollect=True, savename=None):
    if recollect:
        master_df = _recollect_data_df(filepath=filepath, fill_missing_xyzpositions=fill_missing_xyzpositions, result=result, all_columns=all_columns, cell_init_options=cell_init_options,
                                       settings_options=settings_options, opsin_options=opsin_options, field_options=field_options, positional_input=positional_input, savename=savename, save_recollect=save_recollect)
    else:
        master_df = pd.read_csv(os.path.join(filepath, filename), index_col=0)
    return master_df


def _recollect_data_df(*, filepath, fill_missing_xyzpositions, result, all_columns, cell_init_options, settings_options, opsin_options, field_options, positional_input, savename, save_recollect, exclude_folders_with='202302'):
    if savename is None:
        savename = os.path.join(filepath, 'all_data.csv')
    if os.path.exists(savename):
        savename = '(2).'.join(savename.rsplit('.', 1))

    # list all data and input file locations
    alldata_files = [x for x in glob.glob(os.path.join(
        filepath, '**/data.json'), recursive=True) if not ('ToConcat' in x or exclude_folders_with in x)]
    allinput_files = [x for x in glob.glob(os.path.join(
        filepath, '**/input.json'), recursive=True) if not ('ToConcat' in x or exclude_folders_with in x)]
    dir_list = [x for x in glob.glob(os.path.join(
        filepath, '*')) if os.path.isdir(x) and not ('ToConcat' in x or exclude_folders_with in x)]
    print(
        f"result files: {len(alldata_files)}, input files: {len(allinput_files)}, directories: {len(dir_list)}")

    # init master_df: dataframe where all results will be stored
    master_df = pd.DataFrame(columns=all_columns)
    idx = -1
    master_dict = {}
    for i, (data_path, input_path) in enumerate(zip(alldata_files, allinput_files)):
        print(i, '/', len(alldata_files), end='\r')
        # Load data and input file
        with open(data_path, 'r') as f:
            mydata = json.load(f)
        with open(input_path, 'r') as f:
            myinput = json.load(f)

        # store global input info
        cell_init_dict = {'neurontemplate': myinput['info']['settings']['cellsopt']['neurontemplate'], **{key: myinput['info']['settings']['cellsopt']['init_options'][key.rsplit(
            '_', 1)[0]] for key in cell_init_options[:3]}, **{key: myinput['info']['settings']['cellsopt']['cellTrans_options']['rt'][i] for i, key in enumerate(cell_init_options[4:])}}
        settings_dict = {
            key: myinput['info']['settings'][key] for key in settings_options}
        opsin_dict = {**{opsin_options[0]: myinput['info']['settings']['cellsopt']['opsin_options'][opsin_options[0]+'_total']}, **{
            key: myinput['info']['settings']['cellsopt']['opsin_options'][key] for key in opsin_options[1:]}}
        field_dict = {field_options[0]: myinput['info']['settings']['stimopt']['Ostimparams']['filepath'].rsplit('/', 1)[-1].rsplit('.txt', 1)[0],
                      field_options[1]: myinput['info']['settings']['analysesopt']['SDOptogenx']['r_p'.join(field_options[1].split('P'))+'OI']}

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
        durs = myinput['info']['settings']['analysesopt']['SDOptogenx']["durs"]
        for k in mydata.keys():
            cellPos_dict = {**{key: myinput[k]['xT'][i]for i, key in enumerate(
                positional_input[0:3])}, **{key: myinput[k][key]for key in positional_input[3:]}}
            for amp, sR, ichr2, dur in zip(*mydata[k]["SDcurve"]["Optogenx"], durs):
                result_dict = {result[0]: amp, result[1]: sR, result[2]: ichr2['abs_1']
                               ['total'], result[3]: ichr2['abs_1']['total_g'], result[-1]: dur}
                idx += 1
                master_dict[idx] = {**result_dict, **cellPos_dict, **info_dict}

    master_df = pd.concat([master_df, pd.DataFrame.from_dict(
        master_dict, orient='index')])
    master_df = master_df.reset_index(drop=True)

    if save_recollect:
        master_df.to_csv(savename)
    return master_df


def fill_missing_xyzpositions(master_df, *, cell_init_options, settings_options, opsin_options, field_options, savename=None, save_flag=True):
    if savename is None:
        savename = os.path.join(filepath, 'all_data.csv')
    savename = savename.rsplit('.csv')[0]+'_filled.csv'
    if os.path.exists(savename):
        savename = '(2).'.join(savename.rsplit('.', 1))
    # append nan missing positions
    setting_keys = ['dur', *cell_init_options, *
                    settings_options, *opsin_options, *field_options]
    master_df['settings_str'] = master_df.apply(
        lambda x: '_'.join([str(x[key]) for key in setting_keys]), axis=1)
    unique_values_columns = {key: np.array(
        master_df[key].unique()) for key in master_df.columns}
    print(len(master_df))

    master_df['theta_0'] = np.round(
        master_df['theta_0'].values.astype(float), 4)
    unique_values_columns['theta_0'] = np.unique(
        np.round(unique_values_columns['theta_0'].astype(float), 4))
    target = []
    for theta_0 in unique_values_columns['theta_0']:
        idx = master_df['theta_0'] == theta_0
        xX, yY, zZ = np.meshgrid(
            master_df['x'].loc[idx].unique().astype(float), master_df['y'].loc[idx].unique().astype(float), master_df['z'].loc[idx].unique().astype(float))
        target.append(np.array((xX.ravel(), yY.ravel(), zZ.ravel())).T)

    usettings_str = list(master_df['settings_str'].unique())
    print(len(usettings_str))
    master_df.head()
    countr = -1
    for uset in usettings_str:
        countr += 1
        print(f"{countr}/{len(usettings_str)}", end='\r')
        intm_df = master_df[master_df['settings_str'] == uset]
        xyztarget = target[np.where(
            unique_values_columns['theta_0'] == intm_df['theta_0'].unique())[0][0]]
        source = np.array((intm_df['x'], intm_df['y'], intm_df['z'])).T
        # source[:,None]: Nx3 -> Nx1x3 (same as source[:,None,:])
        # target==source[:,None]: Mx3 == Nx1x3 -> 1xMx3 == Nx1x3 -> NxMx3
        # np.all(target==source[:, None], axis=2): NxMx3 -> NxM
        # *.any(axis=0): NxM -> M if column (Mi) has no true -> any false -> ~any => true = missing row
        missing = xyztarget[~np.all(
            xyztarget == source[:, None], axis=2).any(axis=0)]
        if len(missing) > 0:
            for i, missing_row in enumerate(missing):
                a = intm_df.iloc[0:1].copy()
                a['x'], a['y'], a['z'] = missing_row
                a['amp'] = np.nan
                insert_position = intm_df.index[-1]+(i+1)/xyztarget.shape[0]
                master_df.loc[insert_position] = a.to_numpy()[0]
    intm_df.head()
    print(len(master_df))
    master_df.tail()
    master_df = master_df.sort_index().reset_index(drop=True)

    if save_flag:
        master_df.to_csv(savename)
    return master_df


def create_vta_df(*, master_df, columns_vta_df, levels, sortkey, usettings_values, nan_tolerance_percentage=0.95, save_flag=True, savepath=None):
    # collect VTA dataframe
    # in mm3

    vta_df = pd.DataFrame(columns=columns_vta_df)
    idx = -1
    vta_dict = {}
    for i, uset in enumerate(usettings_values):
        print(i, '/', len(usettings_values), end='\r')
        intm_df = master_df[master_df[sortkey] == uset].copy()
        X = np.array(intm_df['x'])/1000
        Y = np.array(intm_df['y'])/1000
        Z = np.array(intm_df['z'])/1000
        uX = np.unique(X)
        uY = np.unique(Y)
        uZ = np.unique(Z)
        data = np.array(intm_df['amp'])/1000
        if sum(np.isnan(data))/len(data) < nan_tolerance_percentage:
            order = EcF.checkGridOrder(X, Z)
            if order == 'xy':
                n_row = len(uZ)
                n_col = len(uX)
                X = np.reshape(X, (n_row, n_col))
                Y = np.reshape(Y, (n_row, n_col))
                Z = np.reshape(Z, (n_row, n_col))
                data = np.reshape(data, (n_row, n_col))
            else:
                n_row = len(uX)
                n_col = len(uZ)
                X = np.reshape(X, (n_row, n_col))
                Y = np.reshape(Y, (n_row, n_col))
                Z = np.reshape(Z, (n_row, n_col))
                data = np.reshape(data, (n_row, n_col))

            vta = VTA2D(X, Z, data, intensity=levels, grid_order=order)
        else:
            vta = np.full(levels.shape, np.nan)

        intm_df = intm_df.drop(
            ['x', 'y', 'z', 'amp', 'ichr2', 'gchr2', 'sR'], axis=1)
        intm_df = intm_df.iloc[0:1]
        for vta_i, level_i in zip(vta, levels):
            idx += 1
            intm_df['vta'] = vta_i
            intm_df['level'] = level_i
            vta_dict[idx] = intm_df.to_dict(orient='list')
            for k, v in vta_dict[idx].items():
                vta_dict[idx][k] = v[0]
            #vta_df = pd.concat([vta_df, intm_df])

    vta_df = pd.concat([vta_df, pd.DataFrame.from_dict(
        vta_dict, orient='index')])
    vta_df = vta_df.reset_index(drop=True)
    if save_flag:
        vta_df.to_csv(savepath)
    return vta_df


def load_vta_df(filepath, filename, *, recollect, **create_vta_df_keys):
    if recollect:
        vta_df = create_vta_df(**create_vta_df_keys)
    else:
        vta_df = pd.read_csv(os.path.join(filepath, filename), index_col=0)
    return vta_df


if __name__ == '__main__':
    recollect = False
    fill_missing_xyzpositions_flag = False

    # Load data
    filepath = './Results\SDC\SDC_singlePulse_Ugent470_gray_invivo_multicell'
    filename = 'all_data.csv'

    # list parameters of interest
    cell_init_options = ['phi_0', 'theta_0', 'psi_0',
                         'neurontemplate', 'x_0', 'y_0', 'z_0']
    settings_options = ['seed', 'celsius', 'dt']
    opsin_options = ['Gmax', 'distribution', 'opsinmech',
                     'distribution_method', 'opsinlocations']
    field_options = ['field', 'nPulse']
    positional_input = ['x', 'y', 'z', 'phi', 'theta', 'psi']
    result = ['amp', 'sR', 'ichr2', 'gchr2', 'dur']
    all_columns = result+field_options+opsin_options + positional_input + \
        cell_init_options+settings_options

    master_df = load_data_df(filepath=filepath, filename=filename, recollect=recollect, result=result, all_columns=all_columns, cell_init_options=cell_init_options,
                             settings_options=settings_options, opsin_options=opsin_options, field_options=field_options, positional_input=positional_input, save_recollect=True, savename=None)

    if fill_missing_xyzpositions_flag:
        master_df = fill_missing_xyzpositions(master_df, cell_init_options=cell_init_options, settings_options=settings_options,
                                              opsin_options=opsin_options, field_options=field_options, save_flag=True)

    unique_values_columns = {
        key: master_df[key].unique() for key in all_columns}
    print(master_df.head())

    setting_keys = ['dur', *cell_init_options, *
                    settings_options, *opsin_options, *field_options]
    master_df['settings_str'] = master_df.apply(
        lambda x: '_'.join([str(x[key]) for key in setting_keys]), axis=1)
    columns_vta_df = ['vta', 'dur', 'level', *opsin_options,
                      *field_options, *cell_init_options, *settings_options, ]
    levels = np.logspace(-1, 3, 5)
    usettings_str = list(master_df['settings_str'].unique())
    savepath = os.path.join(filepath, f'vta_logspace(-1,3,5).csv')
    create_vta_df(master_df=master_df, columns_vta_df=columns_vta_df, levels=levels, sortkey='settings_str',
                  usettings_values=usettings_str, nan_tolerance_percentage=0.95, save_flag=True, savepath=savepath)
