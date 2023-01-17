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

# opsinlocations to single word
opsinLocation_map = {'1000': 'soma', '0100': 'axon', '0010': 'apic',
                     '0001': 'dend', '0011': 'alldend', '1111': 'allsec'}


def load_data_df(*, filepath, filename, recollect, result, all_columns, positional_input, cell_init_options, settings_options, opsin_options, field_options, fill_missing_xyzpositions=True, save_recollect=True, savename=None):
    if recollect:
        master_df = _recollect_data_df(filepath=filepath, fill_missing_xyzpositions=fill_missing_xyzpositions, result=result, all_columns=all_columns, positional_input=positional_input,
                                       cell_init_options=cell_init_options, settings_options=settings_options, opsin_options=opsin_options, field_options=field_options, savename=savename, save_recollect=save_recollect)
    else:
        master_df = pd.read_csv(os.path.join(filepath, filename), index_col=0)
    return master_df


def _recollect_data_df(*, filepath, fill_missing_xyzpositions, result, all_columns, positional_input, cell_init_options, settings_options, opsin_options, field_options, savename, save_recollect):
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
        cell_init_dict = {key: myinput['info']['settings']['cellsopt']['init_options'][key.rsplit(
            '_', 1)[0]] for key in cell_init_options}
        settings_dict = {
            key: myinput['info']['settings'][key] for key in settings_options}
        opsin_dict = {**{opsin_options[0]: myinput['info']['settings']['cellsopt']['opsin_options'][opsin_options[0]+'_total']}, **{
            key: myinput['info']['settings']['cellsopt']['opsin_options'][key] for key in opsin_options[1:]}}
        field_dict = {field_options[0]: myinput['info']['settings']['stimopt']['Ostimparams']['filepath'].rsplit('/', 1)[-1].rsplit('.txt', 1)[0],
                      field_options[1]: myinput['info']['settings']['analysesopt']['SDOptogenx']['options'][field_options[1]+'_sdc'],
                      field_options[2]: myinput['info']['settings']['analysesopt']['SDOptogenx']['r_p'.join(field_options[2].split('P'))+'OI'],
                      field_options[3]: myinput['info']['settings']['stimopt']['Ostimparams']['options'][field_options[3]]}

        opsin_dict['opsinlocations'] = opsinLocation_map[''.join(
            [str(int(any([loc in oloc for oloc in opsin_dict['opsinlocations']]))) for loc in list(opsinLocation_map.values())[:4]])]

        info_dict = {**field_dict, **opsin_dict,
                     **cell_init_dict, **settings_dict}

        # store simulation specific info and add to dataframe
        durs = myinput["info"]['settings']['analysesopt']['SDOptogenx']["durs"]
        for k in mydata.keys():
            cellPos_dict = {**{key: myinput[k]['xT'][i]for i, key in enumerate(
                positional_input[0:3])}, **{key: myinput[k][key]for key in positional_input[3:]}}
            for amp, dur in zip(mydata[k]["SDcurve"]["Optogenx"], durs):
                result_dict = {result[0]: amp, result[1]: dur}
                intm = pd.DataFrame(
                    {**result_dict, **cellPos_dict, **info_dict}, index=[0], columns=all_columns)
                master_df = pd.concat([master_df, pd.DataFrame(
                    {**result_dict, **cellPos_dict, **info_dict}, index=[0], columns=all_columns)])
        master_df = master_df.reset_index(drop=True)

    if save_recollect:
        master_df.to_csv(savename)

    if fill_missing_xyzpositions:
        # append nan missing positions
        setting_keys = ['dur', *cell_init_options, *
                        settings_options, *opsin_options, *field_options]
        master_df['settings_str'] = master_df.apply(
            lambda x: '_'.join([str(x[key]) for key in setting_keys]), axis=1)

        print(len(master_df))
        xX, yY, zZ = np.meshgrid(
            unique_values_columns['x'], unique_values_columns['y'], unique_values_columns['z'])
        target = np.array((xX.ravel(), yY.ravel(), zZ.ravel())).T
        print(target.shape)
        usettings_str = list(master_df['settings_str'].unique())
        print(len(usettings_str))
        master_df.head()
        for uset in usettings_str:
            intm_df = master_df[master_df['settings_str'] == uset]

            source = np.array((intm_df['x'], intm_df['y'], intm_df['z'])).T
            # source[:,None]: Nx3 -> Nx1x3 (same as source[:,None,:])
            # target==source[:,None]: Mx3 == Nx1x3 -> 1xMx3 == Nx1x3 -> NxMx3
            # np.all(target==source[:, None], axis=2): NxMx3 -> NxM
            # *.any(axis=0): NxM -> M if column (Mi) has no true -> any false -> ~any => true = missing row
            missing = target[~np.all(
                target == source[:, None], axis=2).any(axis=0)]
            if len(missing) > 0:
                for i, missing_row in enumerate(missing):
                    a = intm_df.iloc[0:1].copy()
                    a['x'], a['y'], a['z'] = missing_row
                    a['amp'] = np.nan
                    insert_position = intm_df.index[-1]+(i+1)/target.shape[0]
                    master_df.loc[insert_position] = a.to_numpy()[0]
        intm_df.head()
        print(len(master_df))
        master_df.tail()
        master_df = master_df.sort_index().reset_index(drop=True)
        savename = savename.rsplit('.csv')[0]+'_filled.csv'
        if save_recollect:
            master_df.to_csv(savename)
    return master_df


def create_vta_df(*master_df, columns_vta_df, levels, sortkey, usettings_values, nan_tolerance_percentage=0.95, save_flag=True, savepath=None):
    # collect VTA dataframe
    # in mm3

    vta_df = pd.DataFrame(columns=columns_vta_df)
    for i, uset in enumerate(usettings_values):
        # print(i)
        intm_df = master_df[master_df[sortkey] == uset].copy()
        X = np.array(intm_df['x'])/1000
        Y = np.array(intm_df['y'])/1000
        Z = np.array(intm_df['z'])/1000
        uX = np.unique(X)
        uY = np.unique(Y)
        uZ = np.unique(Z)
        data = np.array(intm_df['amp'])/1000
        if sum(np.isnan(data))/len(data) < nan_tolerance_percentage:
            order = EcF.checkGridOrder(Y, Z)
            if order == 'xy':
                n_row = len(uZ)
                n_col = len(uY)
                X = np.reshape(X, (n_row, n_col))
                Y = np.reshape(Y, (n_row, n_col))
                Z = np.reshape(Z, (n_row, n_col))
                data = np.reshape(data, (n_row, n_col))
            else:
                n_row = len(uY)
                n_col = len(uZ)
                X = np.reshape(X, (n_row, n_col))
                Y = np.reshape(Y, (n_row, n_col))
                Z = np.reshape(Z, (n_row, n_col))
                data = np.reshape(data, (n_row, n_col))

            vta = VTA2D(Y, Z, data, intensity=levels, grid_order=order)
        else:
            vta = np.full(levels.shape, np.nan)

        intm_df = intm_df.drop(['x', 'y', 'z', 'amp'], axis=1)
        intm_df = intm_df.iloc[0:1]
        for vta_i, level_i in zip(vta, levels):
            intm_df['vta'] = vta_i
            intm_df['level'] = level_i
            vta_df = pd.concat([vta_df, intm_df])

    vta_df = vta_df.reset_index(drop=True)
    if save_flag:
        vta_df.to_csv(savepath)
    return vta_df

def load_vta_df(filepath, filename, *,recollect, **create_vta_df_keys):
    if recollect:
        vta_df = create_vta_df(**create_vta_df_keys)
    else:
        vta_df = pd.read_csv(os.path.join(filepath, filename), index_col=0)
    return vta_df

if __name__ == '__main__':
    recollect = False
    fill_missing_xyzpositions = False
    consider_fill = True

    # Load data
    filepath = './Results/SDC/SDC_CA1PCcACsig5_invivoUgent'
    filename = 'all_data_filled.csv'

    # list parameters of interest
    positional_input = ['x', 'y', 'z', 'phi', 'theta', 'psi']
    cell_init_options = ['phi_0', 'theta_0', 'psi_0']
    settings_options = ['seed', 'celsius', 'dt']
    opsin_options = ['Gmax', 'distribution', 'opsinmech',
                     'distribution_method', 'opsinlocations']
    field_options = ['field', 'dc', 'nPulse', 'prf']
    result = ['amp', 'dur']
    all_columns = result+field_options+opsin_options + \
        cell_init_options+positional_input+settings_options

    master_df = load_data_df(filepath=filepath, filename=filename, recollect=recollect, result=result, all_columns=all_columns, positional_input=positional_input, cell_init_options=cell_init_options,
                             settings_options=settings_options, opsin_options=opsin_options, field_options=field_options, fill_missing_xyzpositions=True, save_recollect=True, savename=None)

    unique_values_columns = {
        key: master_df[key].unique() for key in all_columns}
    print(master_df.head())

    columns_vta_df = ['vta', 'dur', 'level', *opsin_options,
                      *field_options, *cell_init_options, *settings_options, ]
    levels = np.logspace(-1, 3, 5)
    usettings_str = list(master_df['settings_str'].unique())
    savepath = os.path.join(filepath, f'vta_logspace(-1,3,5).csv')
    create_vta_df(master_df, columns_vta_df, levels, sortkey='settings_str',
                  usettings_values=usettings_str, nan_tolerance_percentage=0.95, save_flag=True, savepath=savepath)
