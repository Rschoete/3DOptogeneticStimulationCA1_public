from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.ndimage import binary_dilation, binary_erosion


def selection_generator(df: pd.DataFrame, unique_values_columns: dict, verbose: bool = False, **kwargs) -> np.ndarray:
    idx = np.ones(len(df)).astype(bool)
    for key, val in kwargs.items():
        if verbose:
            print(
                f"{key}: {val} in unique_values_columns: {val in unique_values_columns[key]}")
        idx = idx & (df[key] == val)
    return idx


def VTA2D_count_axialsym(rR: np.ndarray, zZ: np.ndarray, data: np.ndarray, *, intensity: list, gridorder: bool):
    if gridorder == 'ij':
        # make meshgrid order xy
        #gridorder = 'ij'
        rR = rR.T
        zZ = zZ.T
        data = data.T
        #gridorder = 'xy'
    dZ = np.unique(np.round(np.diff(zZ, axis=0).ravel(), 6))
    dR = np.unique(np.round(np.diff(rR, axis=1).ravel(), 6))
    if len(dZ) > 1 or len(dR) > 1:
        raise ValueError('method currently only for fixed dZ or dR')
    dZs = np.ones((zZ.shape[0], 1))*dZ
    dZs[0, :] = dZ/2
    dZs[-1, :] = dZ/2
    vols = dZs * np.pi * \
        np.hstack((np.zeros((zZ.shape[0], 1)), np.diff(rR**2, axis=1)))

    # create dilated volumes
    dZs = np.ones((zZ.shape[0]+2, 1))*dZ
    dZs[0, :] = dZ/2
    dZs[-1, :] = dZ/2
    rR2 = np.hstack((rR, rR[:, -1:]+dR))
    rR2 = np.vstack((rR2[0:1, :], rR2, rR2[-1:, :]))
    vols_dilated = dZs * np.pi * \
        np.hstack((np.zeros((zZ.shape[0]+2, 1)), np.diff(rR2**2, axis=1)))

    vta_lower = []
    vta_upper = []
    for intens in intensity:
        mask = data <= intens
        mask_dilated = np.zeros((mask.shape[0]+2, mask.shape[1]+1))
        idx = np.where(mask)
        mask_dilated[idx[0]+1, idx[1]] = 1
        mask_dilated = binary_dilation(mask_dilated)

        vols_masked = vols[mask]
        vta_lower.append(sum(vols_masked))
        vols_masked_dilated = vols_dilated[mask_dilated]
        vta_upper.append(sum(vols_masked_dilated))
    return vta_lower, vta_upper


def SURFVTA2D_count_hard_lower_upper(xX: np.ndarray, zZ: np.ndarray, data: np.ndarray, *, intensity: list, gridorder: str, radial_data: bool = False):
    # calculation of lower and upper surfaces
    # lower with erosion first
    # upper with dilation
    # logic average provided as well
    if gridorder == 'ij':
        # make meshgrid order xy
        #gridorder = 'ij'
        xX = xX.T
        zZ = zZ.T
        data = data.T
        #gridorder = 'xy'

    dZ = np.unique(np.round(np.diff(zZ, axis=0).ravel(), 6))
    dX = np.unique(np.round(np.diff(xX, axis=1).ravel(), 6))
    if len(dZ) > 1 or len(dX) > 1:
        raise ValueError('method currently only for fixed dZ or dX')
    dZs = np.ones((zZ.shape[0], 1))*dZ
    dZs[0, :] = 0
    dXs = np.ones((1, xX.shape[1]))*dX
    dXs[:, 0] = 0
    surfs_opening = dZs*dXs

    # create dilated volumes
    dZs = np.ones((zZ.shape[0]+2, 1))*dZ
    dZs[0, :] = 0
    dXs = np.ones((1, xX.shape[1]+2-1*radial_data))*dX
    dXs[:, 0] = 0
    surfs_dilated = dZs*dXs

    # create avg expected surf
    dZs = np.ones((zZ.shape[0], 1))*dZ
    dXs = np.ones((1, xX.shape[1]))*dX
    if radial_data:
        dXs[:, 0] = dX/2
    surfs_avg = dZs*dXs

    surf_lower = []
    surf_upper = []
    surf_avg = []
    for intens in intensity:
        mask = data <= intens
        mask_dilated = np.zeros(
            (mask.shape[0]+2, mask.shape[1]+2-1*radial_data))
        idx = np.where(mask)
        mask_dilated[idx[0]+1, idx[1]+1-radial_data] = 1
        mask_dilated = _enclosed_by_fourTrue(binary_dilation(
            mask_dilated, structure=np.ones((3, 3)))) & _enclosed_by_atleastOne(mask_dilated)

        mask_opening = _enclosed_by_fourTrue(mask)

        surfs_masked_opened = surfs_opening[mask_opening.astype(bool)]
        surf_lower.append(sum(surfs_masked_opened)*(1+radial_data))

        surfs_masked_dilated = surfs_dilated[mask_dilated]
        surf_upper.append(sum(surfs_masked_dilated)*(1+radial_data))

        surfs_masked = surfs_avg[mask]
        surf_avg.append(sum(surfs_masked)*(1+radial_data))
    return surf_avg, surf_lower, surf_upper


def SURFVTA2D_count_soft_lower_upper(xX: np.ndarray, zZ: np.ndarray, data: np.ndarray, *, intensity: list, gridorder: str, radial_data: bool = False):
    if gridorder == 'ij':
        # make meshgrid order xy
        #gridorder = 'ij'
        xX = xX.T
        zZ = zZ.T
        data = data.T
        #gridorder = 'xy'
    dZ = np.unique(np.round(np.diff(zZ, axis=0).ravel(), 6))
    dX = np.unique(np.round(np.diff(xX, axis=1).ravel(), 6))
    if len(dZ) > 1 or len(dX) > 1:
        raise ValueError('method currently only for fixed dZ or dX')
    dZs = np.ones((zZ.shape[0], 1))*dZ
    dZs[0, :] = dZ/2
    dZs[-1, :] = dZ/2
    dXs = np.ones((1, xX.shape[1]))*dX
    dXs[:, 0] = 0
    surfs = dZs*dXs

    # create dilated volumes
    dZs = np.ones((zZ.shape[0]+2, 1))*dZ
    dZs[0, :] = dZ/2
    dZs[-1, :] = dZ/2
    dXs = np.ones((1, xX.shape[1]+2-1*radial_data))*dX
    dXs[:, 0] = 0
    surfs_dilated = dZs*dXs

    surf_lower = []
    surf_upper = []
    for intens in intensity:
        mask = data <= intens
        mask_dilated = np.zeros(
            (mask.shape[0]+2, mask.shape[1]+2-1*radial_data))
        idx = np.where(mask)
        mask_dilated[idx[0]+1, idx[1]+1-radial_data] = 1
        mask_dilated = binary_dilation(mask_dilated)
        # for lower bound only retain columns if two occurances in a row (this way calculate same when not radial als when radial data)
        mask = mask*(np.cumsum(mask, axis=1) > 1)
        mask_dilated = mask_dilated*(np.cumsum(mask_dilated, axis=1) > 1)
        # if not radial_data:
        #     mask_dilated = mask_dilated * \
        #         (np.fliplr(np.cumsum(np.fliplr(mask_dilated), axis=1) > 1))

        surfs_masked = surfs[mask]
        surf_lower.append(sum(surfs_masked)*(1+radial_data))
        surfs_masked_dilated = surfs_dilated[mask_dilated]
        surf_upper.append(sum(surfs_masked_dilated)*(1+radial_data))
    return surf_lower, surf_upper


def VTA2D_from_contour_axialsym(rR, zZ, Data, *, grid_order: Literal['xy', 'ij'], cntrs=None, intensity: list, close_plot=True, cntr_order: Literal['rz', 'zr'] = None) -> float:
    vta = []
    fig_generated = False
    if cntrs is None:
        fig, ax = plt.subplots(1, 1)
        cntrs = ax.contour(zZ, rR, Data, intensity)  # cntr_zr
        cntr_order = 'zr'
        fig_generated = True

    else:
        if cntr_order is None:
            raise ValueError('provide cntr_order when cntr provided')

    for cntr in cntrs.collections:
        r = np.array([])
        z = np.array([])
        for path in cntr.get_paths():
            # if single contour consist of various pieces (eg start inside domain moves outside back inside => two different pieces)
            # then concatenate together
            if cntr_order == 'rz':
                r = np.concatenate((r, path.vertices[:, 0]))
                z = np.concatenate((z, path.vertices[:, 1]))
            else:
                r = np.concatenate((r, path.vertices[:, 1]))
                z = np.concatenate((z, path.vertices[:, 0]))
        if len(z) == 0:
            vta.append(np.nan)
        else:
            if (np.min(z) != z[0]) and not grid_order == 'ij':
                r = np.flip(r)
                z = np.flip(z)
            if (np.min(z) != z[0]):
                r = np.roll(r, -z.argmin())
                z = np.roll(z, -z.argmin())
            # print('z:',z)
            # print('r:',r)
            if z[0] > min(zZ.ravel()):
                # fill the gap: if starts at certain z>z_min => contour is bigger than r_space => include cylinder from x_min -> z with r = r_max (still underestim but less: lower bound)
                z = np.append(min(zZ.ravel()), z)
                r = np.append(max(rR.ravel()), r)
            vta.append(np.abs(np.pi*integrate.trapezoid(r**2, z)))

    if fig_generated:
        plt.close(fig)
        del ax, fig, cntrs
    return vta


def best_optrode_position_zdir(xX: np.ndarray, zZ: np.ndarray, data: np.ndarray, data_TAC: np.ndarray, *, intensity: list, gridorder: str):
    if gridorder == 'ij':
        # make meshgrid order xy
        #gridorder = 'ij'
        xX = xX.T
        zZ = zZ.T
        data = data.T
        data_TAC = data_TAC.T
        #gridorder = 'xy'
    bestz = []
    bestz_TAC = []
    worstz_TAC = []
    bestz_TACamp = []
    worstz_TACamp = []
    for intens in intensity:
        mask = data <= intens
        if not np.any(mask.ravel()):
            bestz.append(np.nan)
            bestz_TAC.append(np.nan)
            worstz_TAC.append(np.nan)
            bestz_TACamp.append(np.nan)
            worstz_TACamp.append(np.nan)
            continue

        # find row of interest
        sumMask = np.sum(mask, axis=1)

        # best location
        idx = np.where(sumMask == max(sumMask))[0]
        if len(idx) > 1:
            data_masked = data*mask
            data_TAC_masked = data_TAC*mask

            data_masked = data_masked[idx, :]
            data_masked[np.isnan(data_masked)] = 0

            data_sums = []
            for i in range(data_masked.shape[0]):
                idx_data = np.where(data_masked[i, :])[0]
                data_sums.append(
                    (data_masked[i, idx_data[0]]+data_masked[i, idx_data[-1]]))

            idx2 = np.nanargmin(data_sums)
            #idx2 = np.argmin(np.sum(data_masked,axis=0))
            bestz.append(zZ[idx[idx2], 0])

            # best z TAC
            data_TAC_masked = data_TAC_masked[idx, :]
            data_TAC_masked[np.isnan(data_TAC_masked)] = 0
            data_TAC_avg = np.sum(data_TAC_masked, axis=1) / \
                np.sum(mask[idx, :], axis=1)
            idx_TAC_min = np.nanargmin(data_TAC_avg)
            bestz_TAC.append(zZ[idx[idx_TAC_min], 0])

            data_TACamp_avg = np.nansum(data_TAC_masked/data_masked, axis=1) / \
                np.sum(mask[idx, :], axis=1)
            idx_TACamp_min = np.nanargmin(data_TACamp_avg)
            bestz_TACamp.append(zZ[idx[idx_TACamp_min], 0])

        else:
            bestz.append(zZ[idx[0], 0])
            bestz_TAC.append(zZ[idx[0], 0])
            bestz_TACamp.append(zZ[idx[0], 0])

        # worst location
        idx_worst = np.where(sumMask == min(sumMask))[0]
        if len(idx_worst) > 1:
            # if nan in data => could not find threshold -> above upper limit of simulation therefore worst position
            data_w = data[idx_worst, :]
            if any(np.isnan(data_w).ravel()):
                idx_nan = np.sum(np.isnan(data_w), axis=1)
                idx_nan = np.where(idx_nan == max(idx_nan))[0]
            else:
                idx_nan = np.arange(len(idx_worst)).astype(int)

            data_w = data_w[idx_nan, :]
            data_TAC_w = data_TAC[idx_worst, :]
            data_TAC_w = data_TAC_w[idx_nan, :]
            if any(np.isnan(data_TAC_w.ravel())):
                raise ValueError('data_TAC_w contains nan?')
            data_remaining_nan = np.isnan(data_w)
            data_w[data_remaining_nan] = 0
            TAC_remaining_nan = np.isnan(data_TAC_w)
            data_TAC_w[TAC_remaining_nan] = 0

            data_TAC_avg = np.sum(data_TAC_w, axis=1) / \
                np.sum(~TAC_remaining_nan, axis=1)
            idx_TAC_max = np.nanargmax(data_TAC_avg)
            worstz_TAC.append(zZ[idx_worst[idx_nan[idx_TAC_max]], 0])

            data_TACamp_avg = np.nansum(data_TAC_w/data_w, axis=1) / \
                np.sum((~data_remaining_nan) & (~TAC_remaining_nan), axis=1)
            idx_TACamp_max = np.nanargmax(data_TACamp_avg)
            worstz_TACamp.append(zZ[idx_worst[idx_nan[idx_TACamp_max]], 0])
        else:
            worstz_TAC.append(zZ[idx_worst, 0])
            worstz_TACamp.append(zZ[idx_worst, 0])

    idx_min = np.argmin(data, axis=0)
    bestZ_perR = zZ[idx_min, 0]
    Imin = data[idx_min, np.arange(len(idx_min))]
    return bestz, bestz_TAC, worstz_TAC, bestz_TACamp, worstz_TACamp, bestZ_perR, Imin


def best_optrode_position_xdir(xX: np.ndarray, zZ: np.ndarray, data: np.ndarray, data_TAC: np.ndarray, *, intensity: list, gridorder: str):
    if gridorder == 'ij':
        # make meshgrid order xy
        #gridorder = 'ij'
        xX = xX.T
        zZ = zZ.T
        data = data.T
        data_TAC = data_TAC.T
        #gridorder = 'xy'

    bestx = []
    bestx_TAC = []
    worstx_TAC = []
    for intens in intensity:
        mask = data <= intens
        if not np.any(mask.ravel()):
            bestx.append(np.nan)
            bestx_TAC.append(np.nan)
            worstx_TAC.append(np.nan)
            continue

        # find row of interest
        sumMask = np.sum(mask, axis=0)
        idx = np.where(sumMask == max(sumMask))[0]
        if len(idx) > 1:
            data_masked = data*mask
            data_masked = data_masked[:, idx]
            data_masked[np.isnan(data_masked)] = 0

            data_sums = []
            for i in range(data_masked.shape[1]):
                idx_data = np.where(data_masked[:, i])[0]
                data_sums.append(
                    (data_masked[idx_data[0], i]+data_masked[idx_data[-1], i]))

            idx2 = np.argmin(data_sums)
            #idx2 = np.argmin(np.sum(data_masked,axis=0))
            bestx.append(xX[0, idx[idx2]])

            # best x TAC
            data_TAC_masked = data_TAC*mask
            data_TAC_masked = data_TAC_masked[:, idx]
            data_TAC_masked[np.isnan(data_TAC_masked)] = 0
            data_TAC_avg = np.sum(data_TAC_masked, axis=0) / \
                np.sum(mask[:, idx], axis=0)
            idx_TAC_min = np.nanargmin(data_TAC_avg)
            idx_TAC_max = np.nanargmax(data_TAC_avg)
            bestx_TAC.append(xX[0, idx[idx_TAC_min]])
            worstx_TAC.append(xX[0, idx[idx_TAC_max]])

        else:
            bestx.append(xX[0, idx[0]])
            bestx_TAC.append(xX[0, idx[0]])
            worstx_TAC.append(xX[0, np.argmax(np.max(data_TAC, axis=0))])

    idx_min = np.argmin(data, axis=1)
    bestx_perz = xX[0, idx_min]
    Imin = data[np.arange(len(idx_min)), idx_min]
    return bestx, bestx_TAC, worstx_TAC, bestx_perz, Imin


def _enclosed_by_fourTrue(a):
    # origin:
    #  0  0: [[1, 1], [1, (1)]]
    #  0 -1: [[1, 1], [(1), 1]]
    # -1  0: [[1, (1)], [1, 1]]
    # -1 -1: [[(1), 1], [1, 1]]
    return binary_erosion(a, structure=np.ones((2, 2)), origin=(0, 0)).astype(bool)


def _enclosed_by_atleastOne(a):
    # dilation
    #  0  0: [[(1), 1], [1, 1]]
    #  0 -1: [[1, (1)], [1, 1]]
    # -1  0: [[1, 1], [(1), 1]]
    # -1 -1: [[1, 1], [1, (1)]]
    '''

    see source code in scipy ndimage.binary_dilation (27/02/2023)
        for ii in range(len(origin)):
            origin[ii] = -origin[ii]
            if not structure.shape[ii] & 1:
                origin[ii] -= 1

        return _binary_erosion(input, structure, iterations, mask,
                            output, border_value, origin, 1, brute_force)
    '''
    return binary_dilation(a, structure=np.ones((2, 2)), origin=(-1, -1)).astype(bool)
