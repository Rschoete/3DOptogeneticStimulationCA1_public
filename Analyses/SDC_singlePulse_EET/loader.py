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
from Analyses.SDC_singlePulse_singleOpticField.tools import *

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CODE IS NOT GENERIC DESIGNED TO COLLECT DATA FROM FOLDER WHERE SCRIPT IS LOCATED
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# opsinlocations to single word
opsinLocation_map = {'pc': {'1000': 'soma', '0100': 'axon', '0010': 'apic',
                     '0001': 'dend', '0011': 'alldend', '1111': 'allsec'},
                     'int': {'100': 'soma', '010': 'axon',
                             '001': 'dend', '111': 'allsec'}}


def load_data_df(*, filepath, filename, recollect, result, all_columns, cell_init_options, settings_options, opsin_options, field_options, positional_input, EETsimidx, EET_info, save_recollect=True, savename=None):
    if recollect:
        filepaths = [x for x in glob.glob(
            os.path.join(filepath, '*')) if os.path.isdir(x)]
        master_df = pd.DataFrame(columns=all_columns)
        for fp in filepaths:
            master_df_tmp = _recollect_data_df(filepath=fp, result=result, all_columns=all_columns, cell_init_options=cell_init_options,
                                               settings_options=settings_options, opsin_options=opsin_options, field_options=field_options, positional_input=positional_input, EETsimidx=EETsimidx)
            master_df = pd.concat([master_df, master_df_tmp])
        master_df = append_EETinfo(
            master_df, EET_info, master_key=EETsimidx[0])
        if savename is None:
            savename = os.path.join(filepath, 'all_data.csv')
        if os.path.exists(savename):
            savename = '(2).'.join(savename.rsplit('.', 1))
        master_df.to_csv(savename)
    else:
        master_df = pd.read_csv(os.path.join(filepath, filename), index_col=0)
    return master_df


def _recollect_data_df(*, filepath, result, all_columns, cell_init_options, settings_options, opsin_options, field_options, positional_input, EETsimidx, exclude_folders_with='2023'):

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

        EETsimidx_nr = int(data_path.rsplit(
            '\\data', 1)[0].rsplit('_', 1)[-1])
        info_dict = {**field_dict, **opsin_dict,
                     **cell_init_dict, **settings_dict, EETsimidx[0]: EETsimidx_nr}

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
    return master_df


def fill_missing_xyzpositions(master_df, *, savename=None, save_flag=True):
    if savename is None:
        savename = os.path.join(filepath, 'all_data.csv')
    savename = savename.rsplit('.csv')[0]+'_filled.csv'
    if os.path.exists(savename):
        savename = '(2).'.join(savename.rsplit('.', 1))
    # append nan missing positions
    setting_keys = ['dur', 'theta_0', 'neurontemplate', 'EETsimidx']
    master_df['settings_str'] = master_df.apply(
        lambda x: '_'.join([str(x[key]) for key in setting_keys]), axis=1)
    unique_values_columns = {key: np.array(
        master_df[key].unique()) for key in master_df.columns}
    print(len(np.unique(master_df['settings_str'])), len(master_df), len(master_df)/len(
        np.unique(master_df['settings_str'])))

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
    print(len(master_df))
    master_df = master_df.sort_index().reset_index(drop=True)
    master_df = master_df.drop(['settings_str'], axis=1)
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
        theta = np.unique(np.round(intm_df['theta_0'], 2))

        data = np.array(
            (intm_df['x'], intm_df['z'], intm_df['amp'])).T
        data_TAC = np.array(
            (intm_df['x'], intm_df['z'], intm_df['TAC'])).T
        uX = np.unique(data[:, 0])
        uZ = np.unique(data[:, 1])
        # if ninterp, data_toplot[-1] is in 'ij' order
        data = EcF.prepareDataforInterp(data, 'ninterp', sorted=False)
        data_TAC = EcF.prepareDataforInterp(data_TAC, 'ninterp', sorted=False)
        xX, zZ = np.meshgrid(uX, uZ, indexing='ij')

        if sum(np.isnan(data[-1].ravel()))/len(data[-1].ravel()) < nan_tolerance_percentage:

            if theta != 0:
                vta_low, vta_up = VTA2D_count_axialsym(
                    xX, zZ, data[-1], intensity=levels, gridorder='ij')
                surf_avg, surf_low, surf_up = SURFVTA2D_count_hard_lower_upper(
                    xX, zZ, data[-1], intensity=levels, gridorder='ij', radial_data=True)
            else:
                vta_low = np.full(levels.shape, np.nan)
                vta_up = np.full(levels.shape, np.nan)
                surf_avg, surf_low, surf_up = SURFVTA2D_count_hard_lower_upper(
                    xX, zZ, data[-1], intensity=levels, gridorder='ij', radial_data=False)

            # only use zdir because data is reorganized so that direction of interest is in zdir
            best_optrode_pos, best_optrode_pos_TAC, worst_optrode_pos_TAC, best_optrode_pos_TACamp, worst_optrode_pos_TACamp, *_ = best_optrode_position_zdir(
                xX, zZ, data[-1], data_TAC[-1], intensity=levels, gridorder='ij')
        else:
            vta_low = np.full(levels.shape, np.nan)
            vta_up = np.full(levels.shape, np.nan)
            surf_avg = np.full(levels.shape, np.nan)
            surf_low = np.full(levels.shape, np.nan)
            surf_up = np.full(levels.shape, np.nan)
            best_optrode_pos = np.full(levels.shape, np.nan)
            best_optrode_pos_TAC = np.full(levels.shape, np.nan)
            worst_optrode_pos_TAC = np.full(levels.shape, np.nan)
            best_optrode_pos_TACamp = np.full(levels.shape, np.nan)
            worst_optrode_pos_TACamp = np.full(levels.shape, np.nan)

        intm_df = intm_df.drop(
            ['x', 'y', 'z', 'amp', 'ichr2', 'gchr2', 'sR', 'TAC'], axis=1)
        intm_df = intm_df.drop(
            ['TAC_log10', 'Gmax_log10', 'dur_log10', 'amp_log10', 'nPulse'], axis=1)

        intm_df = intm_df.iloc[0:1]
        for vta_low_i, vta_up_i, surf_avg_i, surf_low_i, surf_up_i, op_pos_i, op_pos_i_TAC, wop_pos_i_TAC, op_pos_i_TACamp, wop_pos_i_TACamp, level_i in zip(vta_low, vta_up, surf_avg, surf_low, surf_up, best_optrode_pos, best_optrode_pos_TAC, worst_optrode_pos_TAC, best_optrode_pos_TACamp, worst_optrode_pos_TACamp, levels):
            idx += 1
            intm_df['vta_low'] = vta_low_i
            intm_df['vta_up'] = vta_up_i
            intm_df['surf_avg'] = surf_avg_i
            intm_df['surf_low'] = surf_low_i
            intm_df['surf_up'] = surf_up_i
            intm_df['level'] = level_i
            intm_df['b_opt_pos'] = op_pos_i
            intm_df['b_opt_pos_TAC'] = op_pos_i_TAC
            intm_df['w_opt_pos_TAC'] = wop_pos_i_TAC
            intm_df['b_opt_pos_TACamp'] = op_pos_i_TACamp
            intm_df['w_opt_pos_TACamp'] = wop_pos_i_TACamp
            vta_dict[idx] = intm_df.to_dict(orient='list')
            # unpack list
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


def _collect_masterdf(filepath, filename, recollect, fill_missing_xyzpositions_flag, EET_info):
    # list parameters of interest
    cell_init_options = ['phi_0', 'theta_0', 'psi_0',
                         'neurontemplate', 'x_0', 'y_0', 'z_0']
    settings_options = ['seed', 'celsius', 'dt']
    opsin_options = ['Gmax', 'distribution', 'opsinmech',
                     'distribution_method', 'opsinlocations']
    field_options = ['field', 'nPulse']
    positional_input = ['x', 'y', 'z', 'phi', 'theta', 'psi']
    EETsimidx = ['EETsimidx']
    result = ['amp', 'sR', 'ichr2', 'gchr2', 'dur']
    all_columns = result+field_options+opsin_options + positional_input + \
        cell_init_options+settings_options+EETsimidx

    master_df = load_data_df(filepath=filepath, filename=filename, recollect=recollect, result=result, all_columns=all_columns, cell_init_options=cell_init_options,
                             settings_options=settings_options, opsin_options=opsin_options, field_options=field_options, positional_input=positional_input, EETsimidx=EETsimidx, EET_info=EET_info, save_recollect=True, savename=None)

    if fill_missing_xyzpositions_flag:
        master_df = fill_missing_xyzpositions(master_df, save_flag=True)
    return master_df


def append_EETinfo(master_df: pd.DataFrame, EET_info: pd.DataFrame, master_key: str = 'EETsimidx', EET_info_key: str = 'sim_idx') -> pd.DataFrame:
    if isinstance(EET_info,dict):
        if not 'celltype' in master_df.colummns:
            master_df['neurontemplate2'] = master_df['neurontemplate'].replace({'CA1_PC_cAC_sig5':'pyr_1', 'CA1_PC_cAC_sig6':'pyr_2', 'cNACnoljp1':'bc_1', 'cNACnoljp2':'bc_2'})
            master_df[['celltype','number']] = master_df['neurontemplate2'].str.split('_',expand=True)
        init_keys = []
        for key,v in EET_info.items():
            idx = master_df['celltype']==key
            columns_EET_info = list(v.columns)
            columns_EET_info.remove(EET_info_key)
            v = v.set_index(EET_info_key)
            for k in columns_EET_info:
                k_tmp = k+'_eet'
                if not k_tmp in init_keys:
                    master_df[k_tmp] = master_df[master_key]
                    init_keys.append(k_tmp)
                master_df.loc[idx,k_tmp] = master_df.loc[idx,k_tmp].map(v[k])
    else:
        columns_EET_info = list(EET_info.columns)
        columns_EET_info.remove(EET_info_key)
        EET_info = EET_info.set_index(EET_info_key)
        for k in columns_EET_info:
            k_tmp = k+'_eet'
            master_df[k_tmp] = master_df[master_key]
            master_df[k_tmp] = master_df[k_tmp].map(EET_info[k])
    return master_df


def _calc_vta(filepath, filename, savepath):
    master_df = pd.read_csv(os.path.join(filepath, filename), index_col=0)
    drop_columns = ['opsinmech', 'distribution',
                    'distribution_method', 'seed', 'field']
    master_df = master_df.drop(drop_columns, axis=1)

    # preprocess master_df
    master_df['amp'] = master_df['amp']/1000  # convert W/m2 -> mW/mm2
    for x in ['x', 'y', 'z']:
        for suffix in ['', '_0']:
            master_df[x+suffix] = master_df[x+suffix]/1000  # convert um -> mm
    master_df['TAC'] = -master_df['ichr2']/master_df['dur'] * \
        1e-5  # convert to uA (ichr2: mA/cm2*um2)
    master_df['Gmax'] = np.round(master_df['Gmax'], 4)
    master_df['theta_0'] = np.round(master_df['theta_0'], 2)
    for x in ['amp', 'Gmax', 'TAC', 'dur']:
        master_df[x+'_log10'] = np.round(np.log10(master_df[x]), 4)
    master_df['neurontemplate'] = master_df['neurontemplate'].replace(
        {'CA1_PC_cAC_sig5': 'pyr_1', 'CA1_PC_cAC_sig6': 'pyr_2', 'cNACnoljp1': 'bc_1', 'cNACnoljp2': 'bc_2'})
    all_columns = master_df.columns

    # change perspective from optrode centered to stratum pyramidale centred
    for theta in master_df['theta_0'].unique():
        if theta < 0:
            label = 'z'
            master_df.loc[master_df['theta_0'] == theta, label] *= -1
            master_df.loc[master_df['theta_0'] == theta, label+'_0'] *= -1
        elif theta == 0:
            label = 'x'
            master_df.loc[master_df['theta_0'] == theta, label] *= -1
            master_df.loc[master_df['theta_0'] == theta, label+'_0'] *= -1
            toz = master_df.loc[master_df['theta_0'] == theta, 'x'].to_numpy()
            tox = master_df.loc[master_df['theta_0'] == theta, 'z'].to_numpy()
            toz_0 = master_df.loc[master_df['theta_0']
                                  == theta, 'x_0'].to_numpy()
            tox_0 = master_df.loc[master_df['theta_0']
                                  == theta, 'z_0'].to_numpy()
            master_df.loc[master_df['theta_0'] == theta, 'x'] = tox
            master_df.loc[master_df['theta_0'] == theta, 'z'] = toz
            master_df.loc[master_df['theta_0'] == theta, 'x_0'] = tox_0
            master_df.loc[master_df['theta_0'] == theta, 'z_0'] = toz_0

        master_df.loc[master_df['theta_0'] == theta, 'z'] = np.round(
            master_df.loc[master_df['theta_0'] == theta, 'z']+master_df.loc[master_df['theta_0'] == theta, 'z_0'], 3)
        master_df.loc[master_df['theta_0'] == theta, 'x'] = np.round(
            master_df.loc[master_df['theta_0'] == theta, 'x']+master_df.loc[master_df['theta_0'] == theta, 'x_0'], 3)

    unique_values_columns_master = {
        key: master_df[key].unique() for key in all_columns}
    for x in ['amp', 'Gmax', 'TAC', 'dur']:
        unique_values_columns_master[x] = np.sort(
            unique_values_columns_master[x])
        unique_values_columns_master[x+'_log10'] = np.sort(
            unique_values_columns_master[x+'_log10'])
    for x in ['x', 'z', 'x_0', 'z_0']:
        unique_values_columns_master[x] = np.sort(
            unique_values_columns_master[x])

    # create sort label
    setting_keys = ['dur', 'theta_0', 'neurontemplate', 'EETsimidx']
    master_df['settings_str'] = master_df.apply(
        lambda x: '_'.join([str(x[key]) for key in setting_keys]), axis=1)

    columns_vta_df = ['vta_low', 'vta_up', 'surf_low', 'surf_up', 'b_opt_pos', 'b_opt_pos_TAC', 'w_opt_pos_TAC', 'b_opt_pos_TACamp', 'w_opt_pos_TACamp', 'dur', 'level', 'theta_0', 'neurontemplate', 'x_0', 'y_0', 'z_0',
                      'Gmax', 'opsinlocations']
    levels = np.logspace(-1, 3, 9)
    usettings_str = list(master_df['settings_str'].unique())
    np.seterr(divide='ignore', invalid='ignore')
    vta_df = create_vta_df(master_df=master_df, columns_vta_df=columns_vta_df, levels=levels, sortkey='settings_str',
                           usettings_values=usettings_str, nan_tolerance_percentage=0.95, save_flag=True, savepath=savepath)
    return vta_df


if __name__ == '__main__':
    recollect = True
    fill_missing_xyzpositions_flag = True

    # Load data
    filepath = "./Results/SDC/SDC_EETUgent470grayinvivo_singlePulse/"
    filename = 'all_data_filled.csv'
    EETinfo = {}
    EETinfo['pyr'] = pd.read_csv(
        './Inputs/samples_EET_multicell_in_opticalField_pitchcelltypesplitpyr_v2.csv')
    EETinfo['bc'] = pd.read_csv(
        './Inputs/samples_EET_multicell_in_opticalField_pitchcelltypesplitint_v2.csv')

    savepath_vta = os.path.join(filepath, f'vta_logspace(-1,3,9).csv')
    master_df = _collect_masterdf(
        filepath, filename, recollect, fill_missing_xyzpositions_flag, EET_info=EETinfo)
    vta_df = _calc_vta(filepath=filepath, filename=filename,
                       savepath=savepath_vta)
