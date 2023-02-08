from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.ndimage import binary_dilation


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


def SURFVTA2D_count(xX: np.ndarray, zZ: np.ndarray, data: np.ndarray, *, intensity: list, gridorder: str, radial_data: bool = False):
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
    dXs[:, 0] = dX/2
    dXs[:, -1] = dX/2
    surfs = dZs*dXs

    # create dilated volumes
    dZs = np.ones((zZ.shape[0]+2, 1))*dZ
    dZs[0, :] = dZ/2
    dZs[-1, :] = dZ/2
    dXs = np.ones((1, xX.shape[1]+2-1*radial_data))*dX
    dXs[:, 0] = dX/2
    dXs[:, -1] = dX/2
    surfs_dilated = dZs*dXs

    surf_lower = []
    surf_upper = []
    for intens in intensity:
        mask = data <= intens
        mask_dilated = np.zeros(
            (mask.shape[0]+2, mask.shape[1]+2-1*radial_data))
        idx = np.where(mask)
        mask_dilated[idx[0]+1, idx[1]+1] = 1
        mask_dilated = binary_dilation(mask_dilated)

        surfs_masked = surfs[mask]
        surf_lower.append(sum(surfs_masked))
        surfs_masked_dilated = surfs_dilated[mask_dilated]
        surf_upper.append(sum(surfs_masked_dilated))
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


def best_optrode_position_axialsym(xX: np.ndarray, zZ: np.ndarray, data: np.ndarray, *, intensity: list, gridorder: str):
    if gridorder == 'ij':
        # make meshgrid order xy
        #gridorder = 'ij'
        xX = xX.T
        zZ = zZ.T
        data = data.T
        #gridorder = 'xy'
    bestz = []
    xX = xX-xX[:, 1]+1  # this is remove X=0 from xX
    for intens in intensity:
        mask = data <= intens
        if not np.any(mask.ravel()):
            bestz.append(np.nan)
            continue
        # find row of interest
        xX_masked = xX*mask
        dX = np.max(xX_masked, axis=1)-np.min(xX_masked, axis=1)
        idx = np.where(dX == max(dX))[0]
        if len(idx) > 0:
            data_masked = data*mask
            data_masked = data_masked[idx, :]

            data_sums = []
            for i in range(data_masked.shape[0]):
                idx_data = np.where(data_masked[i, :])[0]
                data_sums.append(
                    (data_masked[i, idx_data[0]]+data_masked[i, idx_data[-1]]))

            idx2 = np.argmin(data_sums)
            #idx2 = np.argmin(np.sum(data_masked,axis=0))
            bestz.append(zZ[idx[idx2], 0])

        else:
            bestz.append(zZ[idx, 0])
    idx_min = np.argmin(data, axis=0)
    bestZ_perR = zZ[idx_min, 0]
    Imin = data[idx_min, np.arange(len(idx_min))]
    return bestz, bestZ_perR, Imin


def best_optrode_position_xdir(xX: np.ndarray, zZ: np.ndarray, data: np.ndarray, *, intensity: list, gridorder: str):
    if gridorder == 'ij':
        # make meshgrid order xy
        #gridorder = 'ij'
        xX = xX.T
        zZ = zZ.T
        data = data.T
        #gridorder = 'xy'
        bestx = []
        zZ = zZ-zZ[:1, :]+1  # this is remove Z=0 from zZ
    for intens in intensity:
        mask = data <= intens
        if not np.any(mask.ravel()):
            bestx.append(np.nan)
            continue
        # find row of interest
        zZ_masked = zZ*mask
        dZ = np.max(zZ_masked, axis=0)-np.min(zZ_masked, axis=0)
        idx = np.where(dZ == max(dZ))[0]
        if len(idx) > 0:
            data_masked = data*mask
            data_masked = data_masked[:, idx]

            data_sums = []
            for i in range(data_masked.shape[1]):
                idx_data = np.where(data_masked[:, i])[0]
                data_sums.append(
                    (data_masked[idx_data[0], i]+data_masked[idx_data[-1], i]))

            idx2 = np.argmin(data_sums)
            #idx2 = np.argmin(np.sum(data_masked,axis=0))
            bestx.append(xX[0, idx[idx2]])

        else:
            bestx.append(xX[0, idx])

    idx_min = np.argmin(data, axis=1)
    bestx_perz = xX[0, idx_min]
    Imin = data[np.arange(len(idx_min)), idx_min]
    return bestx, bestx_perz, Imin
