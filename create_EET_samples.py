from typing import Literal

import numpy as np
import scipy.stats as sciStat
from SALib.sample import sobol
from SALib.util import _check_groups, compute_groups_matrix, scale_samples


def OAT_sampling_radial_SobolSequence(problem, rN, shift: int = 4, method: Literal['radial', 'trajectory'] = 'radial', base2: bool = True, fast_forward: int = 1, skip_columns=0):
    '''
    based on Camplongo 2011 From screening to quantitative sensitivity analysis. A unified approach
    https://doi.org/10.1016/j.cpc.2010.12.039
    problem dict see SAlib
    rN is number of trajectories
    kN number of parameter values
    '''
    kN = problem['num_vars']
    # first array is zeros, 2 times dimension for a and b
    if base2:
        if not ((rN & (rN-1) == 0)) and rN != 0:
            substr = f'number trajectories not base two: {rN}'
            rN = int(2**np.ceil(np.log2(rN)))
            print(substr+f' set to {rN}')

    RN = rN+shift

    groups = _check_groups(problem)
    if not groups:
        Dg = problem["num_vars"]
    else:
        G, group_names = compute_groups_matrix(groups)
        Dg = len(set(group_names))
        uG, indices, inverse_indices = np.unique(np.sum(
            G*np.arange(0, G.shape[1]), axis=1), return_index=True, return_inverse=True)
        mask = uG[:, None] == inverse_indices[None, :]
        if method == 'trajectory':
            mask = np.cumsum(mask, axis=0)

    sampler = sciStat.qmc.Sobol(d=2*kN+skip_columns, scramble=False)
    # fastforward to exclude zeros
    _ = sampler.fast_forward(fast_forward)
    sobolSequence = sampler.random(RN)
    sobolSequence = sobolSequence[:, skip_columns:]
    a_matrix = sobolSequence[:-shift, :kN]
    b_matrix = sobolSequence[shift:, kN:]

    samples = np.zeros((rN*(Dg+1), kN))

    for i, (a, b) in enumerate(zip(a_matrix, b_matrix)):
        if method == 'radial':
            if Dg != kN:
                tmp = np.vstack(
                    (a[None, :], a[None, :]*(1-mask)+mask*b[None, :]))
            else:
                tmp = a[None, :]*(1-np.diagflat(np.ones(b.shape), -1)
                                  [:, :kN])+np.diagflat(b, -1)[:, :kN]
        elif method == 'trajectory':
            if Dg != kN:
                tmp = np.vstack(
                    (a[None, :], a[None, :]*(1-mask)+mask*b[None, :]))
                print(
                    'WARNING: no random_permutation implemented when trajectory with groups')
            else:
                tmp = a[None, :] * \
                    np.transpose(np.tri(kN, kN+1)) + \
                    b[None, :]*np.tri(kN+1, kN, -1)
                tmp = tmp[:, np.random.permutation(kN)]
        else:
            raise ValueError(method)

        samples[i*(Dg+1):(Dg+1)*(i+1), :] = tmp

    return samples


def merge_sampling_radial_SobolSequence(set1, set2, n1, n2, r):
    nvalues1 = set1.shape[1]
    nvalues2 = set2.shape[1]
    samples = np.zeros((r*(n1+n2+1), nvalues1+nvalues2))
    for i in range(r):
        idx = i*(n1+n2+1)
        idx_set1 = i*(n1+1)
        idx_set2 = i*(n2+1)
        samples[idx, :] = np.concatenate(
            (set1[idx_set1, :], set2[idx_set2, :]))
        for j in range(n1):
            idx += 1
            idx_set1 = i*(n1+1)+j+1
            idx_set2 = i*(n2+1)
            samples[idx, :] = np.concatenate(
                (set1[idx_set1, :], set2[idx_set2, :]))

        for j in range(n2):
            idx += 1
            idx_set1 = i*(n1+1)
            idx_set2 = i*(n2+1)+j+1
            samples[idx, :] = np.concatenate(
                (set1[idx_set1, :], set2[idx_set2, :]))
    return samples


def _main_sample_tissueparam(save=False):
    # could
    problem = {'num_vars': 3, 'groups': ['Group_1', 'Group_2', 'Group_2', ], 'names': [
        'mua', 'mus', 'g'], 'bounds': [[0.42, 0.15*0.42], [11.33, 0.15*11.33], [0.88, 0.03*0.88]], 'dists': ['norm', 'norm', 'norm']}
    x = np.arange(1, 5, 0.01)

    samples = OAT_sampling_radial_SobolSequence(
        problem, 100, method='radial')
    print(samples)
    samples = scale_samples(samples, problem)

    samples = np.hstack((samples, (samples[:, 1]*(1-samples[:, 2]))[:, None]))
    usamples = np.unique(samples, axis=0)
    print(len(samples), len(usamples), np.mean(samples, axis=0), np.std(samples, axis=0),
          np.max(samples, axis=0), np.min(samples, axis=0), )
    fig, axs = plt.subplots(1, 4)
    axs[0].hist(samples[:, 0])
    axs[1].hist(samples[:, 1])
    axs[2].hist(samples[:, 2])
    axs[3].hist(samples[:, 3])
    mydf = pd.DataFrame(samples[:, :3], columns=problem['names'])
    mydf['sim_idx'] = mydf.index
    print(mydf.head())
    if save:
        mydf.to_csv(
            'optical_parameters_graymatter_invivo_EE_oat2.csv', index=False)
    plt.show()


def _main_sample_all_merge_tissue(save=False):
    cells = ['CA1_PC_cAC_sig5', 'CA1_PC_cAC_sig6', 'cNACnoljp1', 'cNACnoljp2']
    locations = ['all', 'soma', 'axon', 'basal']
    pitches = [-np.pi/2, 0, np.pi/2]
    # skip columns in sobol sequence because already used to sample tissue parameters
    skip_columns = 4
    names = ['Gmax', 'cell', 'loc', 'roll', 'pitch']
    bounds = [[1, 0.15], [0, 4], [0, 12], [0, 2*np.pi], [0, 3]]
    dists = ['norm', 'unif', 'unif', 'unif', 'unif']
    problem = {'num_vars': len(names), 'names': names,
               'bounds': bounds, 'dists': dists}

    samples = OAT_sampling_radial_SobolSequence(
        problem, 32, method='radial', skip_columns=skip_columns)
    print(samples)
    samples = scale_samples(samples, problem)
    samples[:, [1, 2, 4]] = np.floor(samples[:, [1, 2, 4]]-1e-6).astype(int)
    samples[samples[:, 1] <= 1, 2] = np.floor(
        samples[samples[:, 1] <= 1, 2]/3).astype(int)
    samples[samples[:, 1] > 1, 2] = np.floor(
        samples[samples[:, 1] > 1, 2]/4).astype(int)

    df = pd.read_csv(
        'Inputs/optical_parameters_graymatter_invivo_EE_oat_idx.csv')
    tissue_Samples = df[['mua', 'mus', 'g']].values
    tissue_Samples = np.hstack(
        (tissue_Samples, (tissue_Samples[:, 1]*(1-tissue_Samples[:, 2]))[:, None]))
    all_samples = merge_sampling_radial_SobolSequence(
        tissue_Samples[:, 0:3], samples, 2, 5, 32)

    # int to categories
    samples_str = list(all_samples.T)
    samples_str[4] = np.array(cells)[samples_str[4].astype(int)]
    samples_str[5] = np.array(locations)[samples_str[5].astype(int)]
    samples_str[7] = np.array(pitches)[samples_str[7].astype(int)]

    usamples = np.unique(all_samples, axis=0)
    print(len(samples), len(usamples), np.mean(samples, axis=0), np.std(samples, axis=0),
          np.max(samples, axis=0), np.min(samples, axis=0), )

    fig, axs = plt.subplots(int(np.ceil(all_samples.shape[1]/4)), 4)
    labels = ['mua', 'mus', 'g']+names
    for i, (label, ax) in enumerate(zip(labels, axs.ravel())):
        ax.hist(samples_str[i])
        ax.set_title(label)

    mydf = pd.DataFrame(list(zip(*samples_str)), columns=labels)
    mydf['sim_idx'] = mydf.index
    print(mydf.head())
    if save:
        mydf.to_csv(
            'samples_EET_multicell_in_opticalField.csv', index=False)
    plt.show()


def _main_sample_pitchcelltypesplit_merge_tissue(save=False):
    cells = ['CA1_PC_cAC_sig5', 'CA1_PC_cAC_sig6', 'cNACnoljp1', 'cNACnoljp2']
    locations = ['all', 'soma', 'axon', 'basal']
    pitches = [-np.pi/2, 0, np.pi/2]
    # skip columns in sobol sequence because already used to sample tissue parameters
    skip_columns = 4
    names = ['Gmax', 'cell', 'loc', 'roll']
    bounds = [[1, 0.15], [0, 2], [0, 12], [0, 2*np.pi]]
    dists = ['norm', 'unif', 'unif', 'unif']
    problem = {'num_vars': len(names), 'names': names,
               'bounds': bounds, 'dists': dists}

    samples = OAT_sampling_radial_SobolSequence(
        problem, 16, method='radial', skip_columns=skip_columns)
    print(samples)
    samples = scale_samples(samples, problem)
    samples[:, [1, 2]] = np.floor(samples[:, [1, 2]]-1e-6).astype(int)

    df = pd.read_csv(
        'Inputs/optical_parameters_graymatter_invivo_EE_oat_idx.csv')
    tissue_Samples = df[['mua', 'mus', 'g']].values
    tissue_Samples = np.hstack(
        (tissue_Samples, (tissue_Samples[:, 1]*(1-tissue_Samples[:, 2]))[:, None]))
    all_samples = merge_sampling_radial_SobolSequence(
        tissue_Samples[:, 0:3], samples, 2, 4, 16)

    usamples = np.unique(all_samples, axis=0)
    print(len(samples), len(usamples), np.mean(samples, axis=0), np.std(samples, axis=0),
          np.max(samples, axis=0), np.min(samples, axis=0), )

    fig, axs = plt.subplots(int(np.ceil(all_samples.shape[1]/4)), 4)
    labels = ['mua', 'mus', 'g']+names
    for i, (label, ax) in enumerate(zip(labels, axs.ravel())):
        ax.hist(all_samples[:, i])
        ax.set_title(label)

    mydf = pd.DataFrame(all_samples, columns=labels)
    mydf['sim_idx'] = mydf.index
    print(mydf.head())
    if save:
        mydf.to_csv(
            'samples_EET_multicell_in_opticalField_pitchcelltypesplit.csv', index=False)
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    # _main_sample_tissueparam()
    _main_sample_pitchcelltypesplit_merge_tissue(save=True)
