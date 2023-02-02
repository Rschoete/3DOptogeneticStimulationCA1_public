from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate


def selection_generator(df: pd.DataFrame, unique_values_columns: dict, verbose: bool = False, **kwargs) -> np.ndarray:
    idx = np.ones(len(df)).astype(bool)
    for key, val in kwargs.items():
        if verbose:
            print(
                f"{key}: {val} in unique_values_columns: {val in unique_values_columns[key]}")
        idx = idx & (df[key] == val)
    return idx


def VTA2D(rR, zZ, Data, *, grid_order: Literal['xy', 'ij'], cntrs=None, intensity: list, close_plot=True, cntr_order: Literal['rz', 'zr'] = None) -> float:
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
