from typing import Literal, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import optimize, stats


# plots
def heatmap(data: np.ndarray, row_labels: list[str], col_labels: list[str], ax=None,
            cbar_kw=None, cbarlabel: str = "", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if not cbar_kw is False:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def heatmap_colorcode(data: np.ndarray, row_labels: list[str], col_labels: list[str], ax=None, cbar_kw=None, cbarlabel: str = "", row_colors: list = None, column_colors: list = None, labelfs: int = 10, lw_grid: int = 1, colorCode_width = 0.01, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels, add color coding at column and rows for extra categorization.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    row_colors = None
        A list or array of length N with the colors of the axis spine for the rows.
    col_colors = None
        A list or array of length N with the colors of the axis spine for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    labelfs = 10
        int: fontsize of the labels
    lw_grid=1
        int: gird line width
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    if row_colors is not None:
        ax_row_cols = ax.inset_axes([-colorCode_width, 0, colorCode_width, 1])
        ax_row_cols.spines[:].set_visible(False)
    if column_colors is not None:
        ax_column_cols = ax.inset_axes([0, 1, 1, colorCode_width,])
        ax_column_cols.spines[:].set_visible(False)

    # Create colorbar
    if not cbar_kw is False:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90,
                           va="bottom", fontsize=labelfs)

    # Show all ticks and label them with the respective list entries.
    if column_colors is None:
        ax.set_xticks(np.arange(data.shape[1]),
                      labels=col_labels, fontsize=labelfs)
        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
                 rotation_mode="anchor")

    else:
        ax_column_cols.imshow(column_colors, aspect='auto')
        ax_column_cols.set_xticks(
            np.arange(data.shape[0]), labels=col_labels, fontsize=labelfs)
        ax_column_cols.set_xticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax_column_cols.set_yticks([])
        ax_column_cols.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks([])
        # Let the horizontal axes labeling appear on top.
        ax_column_cols.tick_params(top=True, bottom=False,
                                   labeltop=True, labelbottom=False)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax_column_cols.get_xticklabels(), rotation=-45, ha="right",
                 rotation_mode="anchor")

    if row_colors is None:
        ax.set_yticks(np.arange(data.shape[0]),
                      labels=row_labels, fontsize=labelfs)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    else:
        ax_row_cols.imshow(row_colors, aspect='auto')
        ax_row_cols.set_yticks(
            np.arange(data.shape[0]), labels=row_labels, fontsize=labelfs)
        ax_row_cols.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax_row_cols.set_xticks([])
        ax_row_cols.tick_params(which="minor", bottom=False, left=False)

        ax.set_yticks([])
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.grid(which="minor", color="w", linestyle='-', linewidth=lw_grid)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def draw_first_diagonal(*args, **kwargs):
    if "ax" in kwargs:
      ax = kwargs['ax']
    else:
      ax = plt.gca()
    xpoints = ypoints = ax.get_xlim()
    ax.plot(xpoints, ypoints, linestyle='--', color='k', lw=1, scalex=False, scaley=False)

def broken_boxplot(df:pd.DataFrame, break_points:tuple[float,float], yaxratio:float=1, fig = None,**boxplot_kwargs):
    '''
    Create broken boxplot
    Input:
        - df: pd.Dataframe
        - break_points: tuple(float,float)
        - yaxratio:float=1, ratio between two pieces
        - fig = None, plt.figure()
        - **boxplot_kwargs
    '''
    if fig is None:
        fig = plt.figure()
    if yaxratio==1:
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=(1, yaxratio))
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
    fig.subplots_adjust(hspace=0.05)  # adjust space between axes

    # plot the same data on both axes
    sns.boxplot(df,ax=ax1,**boxplot_kwargs)
    sns.boxplot(df,ax=ax2,**boxplot_kwargs)
    ylims = ax1.get_ylim()
    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(break_points[1], ylims[1])  # outliers only
    ax2.set_ylim(ylims[0], break_points[0]) # most of the data

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # Now, let's turn towards the cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the axes size or scale
    # Finally, we need to disable clipping.

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)
    return fig, ax1, ax2


# utils
def sort_xy_based_x(x: Union[list, np.ndarray], y: Union[list, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array(x)
    y = np.array(y)
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    return x, y

# statistics
def find_outliers_IQR(df:pd.DataFrame,outlier_distance:float = 1.5) -> tuple[pd.Series, float]:
   '''
   find outliers of a dataframe series and inter quartiel distance
   Input:
    - df: pd.Series
    - outlier_distance:float = 1.5 proportionallity factor on inter quartiel distance for outliers
   '''
   q1=df.quantile(0.25)
   q3=df.quantile(0.75)
   IQR=q3-q1
   outliers = df[((df<(q1-outlier_distance*IQR)) | (df>(q3+outlier_distance*IQR)))].copy()
   return outliers, IQR

def linearReg(dataframe:pd.DataFrame,features:list[str],axs,filter:str = 'amp_log10',xdata_str:str ='Gmax_log10',plot_flag:bool = True,scatterplot_flag:bool=True) -> list[dict]:
    '''
    Perform linear regression on certain features in a dataframe with linregress of scipy.stats

    Input:
    - dataframe:         table with dataset
    - features:          list of column headers to perform linear regression on
    - axs:               list of plt.axes (same length as features)
    - filter:            exclude data points with nan in filter
    - xdata_str:         column header xdata for linreg
    - plot_flag:         create plots of lin reag
    - scatterplot_flag:  include scatterplot of data
    Output:
    - ress:              list of dicts with information of linregress on features
    '''
    ress = []
    tinv = lambda p, df: abs(stats.t.ppf(p/2, df))
    for feat,ax in zip(features,axs):
        xdata = np.array(dataframe[xdata_str])
        ydata = np.array(dataframe[feat])
        idx = ~np.isnan(ydata)
        if filter is not None:
            idx = ~np.isnan(np.array(dataframe[filter])) & idx
        xdata = xdata[idx]
        xvals = np.unique(xdata)
        ydata = ydata[idx]
        res = stats.linregress(xdata, ydata)
        ts = tinv(0.05, len(xdata)-2)
        print(f"\n{feat}\nRsquared: {res.rvalue**2}\nslope (95%): {res.slope:.6f} +/- {ts*res.stderr:.6f}")
        print(f"intercept (95%): {res.intercept:.6f} +/- {ts*res.intercept_stderr:.6f}")


        # calculate MSE
        yestim = xdata*res.slope+res.intercept
        MSE = np.mean((ydata-yestim)**2)

        # calculate own estimate 95% confidence interval
        mymatrix = np.zeros((4,len(xvals)))
        i= -1
        for x in [-1,1]:
            for y in [-1,1]:
                i+=1
                mymatrix[i,:] = xvals*(res.slope+x*ts*res.stderr)+res.intercept+y*ts*res.intercept_stderr
        minslope = np.min(mymatrix,axis=0)
        maxslope = np.max(mymatrix,axis=0)

        #plot

        intm_df = dataframe[idx]
        if plot_flag:
            if scatterplot_flag:
                sns.scatterplot(intm_df,x = xdata_str,y=feat,ax=ax,hue='opsinlocations',style='dur')
                sns.regplot(intm_df,x = xdata_str, y=feat,ax=ax,marker='')
            else:
                sns.regplot(intm_df,x = xdata_str, y=feat,ax=ax)
            ax.plot(xvals,xvals*res.slope+res.intercept)
            ax.plot(xvals,minslope,color='k')
            ax.plot(xvals,maxslope,color='k')
            ax.set_title(feat)
        res = res._asdict()
        res['MSE'] = MSE
        res['Rsquared'] = res['rvalue']**2
        ress.append(res)
    return ress


def calcr2(y:np.ndarray, yestim:np.ndarray, nparams:int, type:float='adjusted') -> float:
    # calculate (adjusted) Rsquared metric
    idx = ~(np.isnan(y) | np.isinf(y))
    y = y[idx]
    yestim = yestim[idx]
    ssres = np.sum((y-yestim)**2)
    sstot = np.sum((y-np.mean(y))**2)
    r2 = 1-ssres/sstot
    if type == 'adjusted':
        r2adj = 1-(1-r2)*(len(y)-1)/(len(y)-nparams)
        return r2adj
    return r2

# curve fit

def Hill_Lap(pd, I0, tau):
    return I0/(1-np.exp(-pd/tau))
def power_ampTAC(TAC, a, b, c):
    return a*TAC**b+c
def loglog_ampTAC(TAC, slope, intercept):
    return slope * np.log10(TAC)+intercept

vHill_Lap = np.vectorize(Hill_Lap)
vpower_ampTAC = np.vectorize(power_ampTAC)
vloglog_ampTAC = np.vectorize(loglog_ampTAC)

def durfit(df, typer2='adjusted'):
    '''
    fit amp and TAC wrt to dur
    '''
    vHill_Lap = np.vectorize(Hill_Lap)
    vpower_ampTAC = np.vectorize(power_ampTAC)
    vloglog_ampTAC = np.vectorize(loglog_ampTAC)

    xdata = df['dur'].to_numpy()
    ydata_TAC = df['TAC'].to_numpy()
    ydata_amp = df['amp'].to_numpy()
    ydata_amp_log10 = df['amp_log10'].to_numpy()
    idx = ~(np.isnan(xdata) | np.isnan(ydata_TAC) | np.isinf(ydata_TAC) | np.isnan(
        ydata_amp) | np.isinf(ydata_amp) | np.isnan(ydata_amp_log10) | np.isinf(ydata_amp_log10))
    xdata = xdata[idx]
    ydata_TAC = ydata_TAC[idx]
    ydata_amp = ydata_amp[idx]
    ydata_amp_log10 = ydata_amp_log10[idx]
    popt_TAC_hl, corr_TAC_hl = optimize.curve_fit(
        Hill_Lap, xdata, ydata_TAC, p0=[np.min(ydata_TAC), 1])
    popt_amp_power, _ = optimize.curve_fit(power_ampTAC, vHill_Lap(
        xdata, *popt_TAC_hl), ydata_amp, p0=[-2, 0.6, 0], bounds=([-10, -10, 0], np.inf))
    popt_amp_hl, _ = optimize.curve_fit(
        Hill_Lap, xdata, ydata_amp, p0=[np.min(ydata_amp), 1])
    # loglog fit amp_log10 vs TAC_log10
    res = stats.linregress(
        np.log10(vHill_Lap(xdata, *popt_TAC_hl)), ydata_amp_log10)

    popt_amp_loglog = [res.slope, res.intercept]
    TAC_estim = vHill_Lap(xdata, *popt_TAC_hl)
    r2_TAC = calcr2(ydata_TAC, TAC_estim, 2, type=typer2)
    r2_amp_power = calcr2(ydata_amp, vpower_ampTAC(
        vHill_Lap(xdata, *popt_TAC_hl), *popt_amp_power), 5, type=typer2)
    r2_amp_loglog = calcr2(ydata_amp_log10, vloglog_ampTAC(
        vHill_Lap(xdata, *popt_TAC_hl), *popt_amp_loglog), 4, type=typer2)
    r2_amp_hl = calcr2(ydata_amp, vHill_Lap(
        xdata, *popt_amp_hl), 2, type=typer2)

    # fig,axs = plt.subplots(2,3)
    # axs[0,0].scatter(np.log10(xdata),np.log10(ydata_TAC))
    # axs[0,0].plot(np.log10(xdata),np.log10(vHill_Lap(xdata,*popt_TAC_hl)))
    # axs[1,0].scatter(np.log10(ydata_TAC),np.log10(vHill_Lap(xdata,*popt_TAC_hl)))

    # axs[0,1].scatter(np.log10(xdata),np.log10(ydata_amp))
    # axs[0,1].plot(np.log10(xdata),np.log10(vpower_ampTAC(vHill_Lap(xdata,*popt_TAC_hl),*popt_amp_power)))
    # axs[0,1].plot(np.log10(xdata),vloglog_ampTAC(vHill_Lap(xdata,*popt_TAC_hl),*popt_amp_loglog))
    # axs[0,1].plot(np.log10(xdata),np.log10(vHill_Lap(xdata,*popt_amp_hl)))
    # axs[1,1].scatter(np.log10(ydata_amp),np.log10(vpower_ampTAC(vHill_Lap(xdata,*popt_TAC_hl),*popt_amp_power)))
    # axs[1,1].scatter(np.log10(ydata_amp),vloglog_ampTAC(vHill_Lap(xdata,*popt_TAC_hl),*popt_amp_loglog))
    # axs[1,1].scatter(np.log10(ydata_amp),np.log10(vHill_Lap(xdata,*popt_amp_hl)))

    # axs[0,2].scatter(np.log10(ydata_TAC),np.log10(ydata_amp))
    # axs[0,2].scatter(np.log10(vHill_Lap(xdata,*popt_TAC_hl)),np.log10(ydata_amp))
    # axs[0,2].plot(np.log10(vHill_Lap(xdata,*popt_TAC_hl)),np.log10(vpower_ampTAC(vHill_Lap(xdata,*popt_TAC_hl),*popt_amp_power)))
    # axs[0,2].plot(np.log10(vHill_Lap(xdata,*popt_TAC_hl)),vloglog_ampTAC(vHill_Lap(xdata,*popt_TAC_hl),*popt_amp_loglog))
    # axs[1,2].scatter(np.log10(ydata_amp),np.log10(vpower_ampTAC(vHill_Lap(xdata,*popt_TAC_hl),*popt_amp_power)))
    # axs[1,2].scatter(np.log10(ydata_amp),vloglog_ampTAC(vHill_Lap(xdata,*popt_TAC_hl),*popt_amp_loglog))

    return popt_TAC_hl, popt_amp_power, popt_amp_hl, popt_amp_loglog, r2_TAC, r2_amp_power, r2_amp_hl, r2_amp_loglog


def Gmaxfitafterdurfit(df, typer2='adjusted', usedur: Literal['loglog', 'power'] = 'power'):
    ''' Two step regression see paper df must contain 'amp_log10', 'TAC', 'dur' and 'Gmax_log10' collumns '''

    vHill_Lap = np.vectorize(Hill_Lap)
    vpower_ampTAC = np.vectorize(power_ampTAC)

    # get durfits
    popt_TAC_hl, popt_amp_power, popt_amp_hl, popt_amp_loglog, r2_TAC, r2_amp_power, r2_amp_hl, r2_amp_loglog = durfit(
        df, typer2=typer2)
    if usedur == 'loglog':
        def amp_log10_fdur(dur): return popt_amp_loglog[0]*np.log10(
            vHill_Lap(dur, *popt_TAC_hl))+popt_amp_loglog[1]
        nparams_fdur = 2+2
        # 10**(popt_amp_loglog[0]*np.log10(vHill_Lap(df['dur'],*popt_TAC_hl))+popt_amp_loglog[1]))
    elif usedur == 'power':
        nparams_fdur = 2+3
        def amp_log10_fdur(dur): return np.log10(
            vpower_ampTAC(vHill_Lap(dur, *popt_TAC_hl), *popt_amp_power))
    else:
        raise ValueError('usedur should be loglog or power')
    vamp_log10_fdur = np.vectorize(amp_log10_fdur)
    df['amp_log10ampfdur_log10'] = df['amp_log10'] - vamp_log10_fdur(df['dur'])

    df['TACdur_log10'] = np.log10(df['TAC']/vHill_Lap(df['dur'], *popt_TAC_hl))
    single_linear_model = smf.ols(
        formula='amp_log10ampfdur_log10 ~ Gmax_log10', data=df).fit()
    p_Gmax_log10 = single_linear_model.params

    def mymodel(dur, Gmax): return amp_log10_fdur(
        dur)+p_Gmax_log10['Gmax_log10']*np.log10(Gmax)+p_Gmax_log10['Intercept']
    vmymodel = np.vectorize(mymodel)
    amp_log10_estim = vmymodel(df['dur'], df['Gmax'])
    r2_tot = calcr2(df['amp_log10'], amp_log10_estim,
                    nparams_fdur+2, type=typer2)
    return vmymodel, r2_tot, single_linear_model, popt_TAC_hl, popt_amp_power, popt_amp_hl, popt_amp_loglog, r2_TAC, r2_amp_power, r2_amp_hl, r2_amp_loglog


# SoFPAn
def estim_surfvta(dur:float, Gmax:float, model:callable, intensity:float, field_axialsymm:np.ndarray, dr:float, dz:float,)->float:
    target_amp_log10 = model(dur, Gmax)
    target_amp = 10**target_amp_log10

    myfield = intensity*field_axialsymm
    above_target = myfield >= target_amp
    surf = np.sum(above_target.ravel()*dr*dz)

    return surf


# EET

def get_importance(mylist):
    'unique gets unique and sorts'
    if all(mylist == 0):
        return np.full(len(mylist), np.nan)
    u, idx = np.unique(mylist, return_inverse=True)
    if max(idx) < (len(mylist)-1):
        idx += len(mylist)-1-max(idx)
    idx_0 = mylist == 0
    idx[idx_0] = 0
    return idx

def EET_indices(r:int, xmin:np.ndarray, xmax:np.ndarray, X:np.ndarray, Y:np.ndarray, design_type:Literal['radial', 'trajectory'], Nboot:int=1, alfa:float=0.05):
    '''
    Compute the sensitivity indices according to the Elementary Effects Test
    (Saltelli, 2008) or 'method of Morris' (Morris, 1991).
    These are: the mean (mi) of the EEs associated to input 'i',
    which measures the input influence; and the standard deviation (sigma)
    of the EEs, which measures its level of interactions with other inputs.

    Basic usage:
    [mi, sigma, EE] = EET_indices(r,xmin,xmax,X,Y,design_type)

    Input:
        r = number of sampling point                           - scalar
        xmin = lower bounds of input ranges                 - vector (1,M)
        xmax = upper bounds of input ranges                 - vector (1,M)
        X = matrix of sampling datapoints where EE must be computed
                                                        - matrix (r*(M+1),M)
        Y = associated output values               - vector (r*(M+1),1)
        design_type = design type (string)
                [options: 'radial','trajectory']
    Output:
        mi = mean of the elementary effects               - vector (1,M)
        sigma = standard deviation of the elementary effects - vector (1,M)
        EE = matrix of elementary effects                 - matrix (r,M)


    Advanced usage:

    [mi, sigma, EE] = EET_indices(r,xmin,xmax,X,Y,design_type,Nboot)
    [mi, sigma, EE] = EET_indices(r,xmin,xmax,X,Y,design_type,Nboot,alfa)

    Optional input:
        Nboot = number of resamples used for boostrapping (default:0)
        alfa = significance level for the confidence intervals estimated
                by bootstrapping (default: 0.05)
    In this case, the output 'mi' and 'sigma' are the mean and standard
    deviation of the EEs averaged over Nboot resamples.

    Advanced usage/2:

    [mi, sigma, EE, mi_sd, sigma_sd, mi_lb, sigma_lb, mi_ub, sigma_ub] = ...
                EET_indices(r,xmin,xmax,X,Y,design_type,Nboot)

    Optional output:
        mi_sd = standard deviation of 'mi' across Nboot resamples
        sigma_sd = standard deviation of 'sigma' across Nboot resamples
        mi_lb = lower bound of 'mi' (at level alfa) across Nboot resamples
        sigma_lb = lower bound of 'sigma' across Nboot resamples
        mi_ub = upper bound of 'mi' (at level alfa) across Nboot resamples
        sigma_ub = upper bound of 'sigma' across Nboot resamples
                                                    - all the above are
                                                    vector (1,M) if Nboot>1
                                                    (empty vector otherwise)
    Or:

    [mi, sigma, EE, mi_sd, sigma_sd, mi_lb, sigma_lb, mi_ub, sigma_ub, ...
        mi_all, sigma_all ] = EET_indices(r,xmin,xmax,X,Y,design_type,Nboot)

    Optional output:
        mi_all = Nboot estimates of 'mi'                    - matrix (Nboot,M)
        sigma_all = Nboot estimates of 'sigma'                 - matrix (Nboot,M)

    REFERENCES:

    Morris, M.D. (1991), Factorial sampling plans for preliminary
    computational experiments, Technometrics, 33(2).

    Saltelli, A., et al. (2008) Global Sensitivity Analysis, The Primer,
    Wiley.

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    bristol.ac.uk/cabot/resources/safe-toolbox/
    '''

    ## Input check
    if not isinstance(r,int):
        raise ValueError("r should be integer")
    if np.ndim(xmin)!=1 or  np.ndim(xmax)!=1:
        raise ValueError("xmax must be a row vector")
    if len(xmin)!=len(xmax):
        raise ValueError("xmin should have same length as xmax")

    Dr = xmax - xmin

    if any(Dr<0):
        raise ValueError("all components of ''xmax'' must be higher than the corresponding ones in xmin")
    M = len(Dr)
    n,m = np.shape(X)
    if n!=r*(M+1):
        raise ValueError("X must have r*(M+1) rows")
    if m!=M:
        raise ValueError("X must have M columns")
    if np.ndim(Y)!=1:
        raise ValueError("Y must be 1 dim array")
    n = np.size(Y)
    if n!=r*(M+1):
        raise ValueError("Y must have r*(M+1) rows")

    ## EET calculation

    M = X.shape[1]
    EE = np.full((r, M), np.nan)  # matrix of elementary effects
    relEE = np.full((r, M), np.nan)  # matrix of elementary effects
    k = 0
    ki = 0
    for i in range(r):
        for j in range(M):
            if design_type == 'radial':  # radial design: EE is the difference
                # between output at one point in the i-th block and output at
                # the 1st point in the block
                dy = abs(Y[k + 1] - Y[ki])
                dx = abs(X[k + 1, j] - X[ki, j])
                EE[i, j] = dy/ dx * Dr[j]
                relEE[i, j] = dy/ dx * Dr[j]/Y[ki]
            elif design_type == 'trajectory':  # trajectory design: EE is the difference
                # between output at one point in the i-th block and output at
                # the previous point in the block (the "block" is indeed a
                # trajectory in the input space composed of points that
                # differ in one component at the time)
                idx = np.where(abs(X[k + 1, :] - X[k, :]) > 0)[0]  # if using 'morris'
                # sampling, the points in the block may not
                # be in the proper order, i.e. each point in the block differs
                # from the previous/next one by one component but we don't know
                # which one; this is here computed and saved in 'idx'
                if len(idx) == 0:
                    raise ValueError(f"X({k},:) and X({k+1},:) are equal")
                if len(idx) > 1:
                    raise ValueError(f"X({k},:) and X({k+1},:) differ in more than one component")
                EE[i, idx] = abs(Y[k + 1] - Y[k]) / abs(X[k + 1, idx] - X[k, idx]) * Dr[idx]
                relEE[i, idx] = abs(Y[k + 1] - Y[k]) / abs(X[k + 1, idx] - X[k, idx]) * Dr[idx]/Y[k]
            else:
                raise ValueError("'design_type' must be one among {'radial','trajectory'}")
            k += 1
        k += 1
        ki = k

    if Nboot > 1:
        bootsize = r
        B = np.floor((np.random.rand(bootsize,Nboot)*r)).astype(int)
        mi_all = np.empty((Nboot,M))
        sigma_all = np.empty((Nboot,M))
        for n in range(Nboot):
            mi_all[n,:] = np.mean(EE[B[:,n],:], axis=0)
            sigma_all[n,:] = np.std(EE[B[:,n],:], axis=0)

        mi = np.mean(mi_all, axis=0)
        mi_sd = np.std(mi_all, axis=0)
        mi_lb = np.sort(mi_all, axis=0)[max(1,round(Nboot*alfa/2))-1,:]
        mi_ub = np.sort(mi_all, axis=0)[round(Nboot*(1-alfa/2))-1,:]

        sigma = np.mean(sigma_all, axis=0)
        sigma_sd = np.std(sigma_all, axis=0)
        sigma_lb = np.sort(sigma_all, axis=0)[max(1,round(Nboot*alfa/2))-1,:]
        sigma_ub = np.sort(sigma_all, axis=0)[round(Nboot*(1-alfa/2))-1,:]

        return mi, sigma, EE, relEE, mi_sd, sigma_sd, mi_lb, sigma_lb, mi_ub, sigma_ub, mi_all, sigma_all
    else:
        mi = np.mean(EE, axis=0)
        sigma = np.std(EE, axis=0)
        return mi, sigma, EE, relEE

def EET_plot(mi,sigma,labels,mi_lb=None,mi_ub=None,sigma_lb=None,sigma_ub=None,clrs=None,cc=None, ax=None, ms=14):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    if clrs is None:
        clrs = list(mpl.colors.TABLEAU_COLORS.values())
    if isinstance(clrs,dict):
        clrs_dict = clrs
        clrs = []
        for l in labels:
            clrs.append(clrs_dict[l])
    # First plot EEs mean & std as circles:
    for m,s,label,clr in zip(mi, sigma, labels,clrs):
        ax.plot(m,s,'ok',markerfacecolor=clr,markersize=ms,markeredgecolor='k',zorder=1, label=label)
    if not (mi_lb is None or mi_ub is None or sigma_lb is None or sigma_ub is None):

        #plot first the larger confidence areas
        size_bounds=mi_ub-mi_lb
        idx = np.argsort(size_bounds)[::-1]

        for i in range(len(idx)):  # add rectangular shade:
            h = ax.fill([mi_lb[idx[i]],mi_lb[idx[i]],mi_ub[idx[i]],mi_ub[idx[i]]],[sigma_lb[idx[i]],sigma_ub[idx[i]],sigma_ub[idx[i]],sigma_lb[idx[i]]],facecolor=clrs[idx[i]],edgecolor='none',zorder=0)

        # Plot again the circles (in case some have been overriden by the rectangles
        # representing confidence bounds)
        for m,s,label,clr in zip(mi, sigma, labels, clrs):
            ax.plot(m,s,'ok',markerfacecolor=clr,markersize=ms,markeredgecolor='k',zorder=1, label=label)

