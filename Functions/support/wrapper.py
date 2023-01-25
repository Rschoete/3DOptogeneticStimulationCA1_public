import matplotlib.pyplot as plt

from .metrics import *
from .plots import *


def AnalysesWrapper(h, input, cell, t, vsoma, traces, ostim_time, ostim_amp, estim_time, estim_amp, aptimevectors, apinfo, idx_sR, amps_SDeVstim, amps_SDoptogenx, pos_VTAeVstim, pos_VTAoptogenx, fig_dir):

    iOptogenx = None
    VTAOptogenx = None
    VTAeVstim = None

    # create colored section plot
    aopt = input.analysesopt
    if aopt.sec_plot_flag:
        print("\t* Section plot")
        ax = cell.sec_plot()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=90, azim=-90)
        if input.save_flag:
            savename = f"{fig_dir}/sec_plot.png"
            plt.savefig(savename)

    # print section positions
    if aopt.print_secpos:
        print("\t* Section positons")
        cell.gather_secpos(print_flag=True)

    # create shapePlots
    print("\t* Shape plots")
    shapePlot(h, cell, aopt.shapeplots, aopt.shapeplot_axsettings,
              input.save_flag and aopt.save_shapeplots, figdir=fig_dir, extension=aopt.shapeplots_extension)

    # recordTotalOptogeneticCurrent
    iOptogenx = calciOptogenx(input, t, traces)

    # succes Ratio
    if aptimevectors is not None:
        succes_ratio = calcSuccesratio(
            input, t, aptimevectors[idx_sR], ostim_time, ostim_amp, estim_time, estim_amp)
    else:
        succes_ratio = None

    # plot recorded traces
    if vsoma is not None:
        print("\t* Plot traces")
        plot_traces(t, vsoma, traces, ostim_time, ostim_amp, estim_time, estim_amp, aopt.tracesplot_axsettings,
                    input.save_flag and aopt.save_traces, figdir=fig_dir, extension=aopt.traces_extension)
    elif not "normal" in input.simulationType:
        print('\t* no traces to plot because "normal" not in input.stimulationType')
    else:
        print("\t* no traces to plot, normally always should plot vsoma???")

    # rastergram
    if aptimevectors is not None:
        print("\t* Raster plot")
        rasterplot(cell, aptimevectors, apinfo, np.array(
            t), input.save_flag and aopt.save_rasterplot, figdir=fig_dir, **aopt.rasterplotopt)
    elif not "normal" in input.simulationType:
        print('\t* no raster plot because "normal" not in input.stimulationType')

    # SDcurve plots
    if amps_SDeVstim is not None:
        SDcopt = input.analysesopt.SDeVstim
        try:
            SDcurveplot(SDcopt, amps_SDeVstim[0], 'V_e stim amp [V]',
                        'eVstim', input.save_flag and aopt.save_SDplot, figdir=fig_dir)
        except Exception as E:
            print(E)

    if amps_SDoptogenx is not None:
        SDcopt = input.analysesopt.SDOptogenx
        try:
            SDcurveplot(SDcopt, amps_SDoptogenx[0], 'Light Intensity stim amp [W/m2]',
                        'optogenx', input.save_flag and aopt.save_SDplot, figdir=fig_dir)
            if SDcopt['record_iOptogenx'] is not None:
                idx = 1 + SDcopt['return_spikeCountperPulse']
                for i in range(sum(['abs' in x for x in amps_SDoptogenx[idx][0].keys()])):
                    SDcurveplot(SDcopt, [x['abs_'+str(i+1)]['total']*-1e-2/dur for x, dur in zip(amps_SDoptogenx[idx], SDcopt.durs)], 'Total opsin current [nA]',
                                'optogenx_current'+str(i+1), input.save_flag and aopt.save_SDplot, figdir=fig_dir)
        except Exception as E:
            print(E)

    if pos_VTAeVstim is not None:
        VTAopt = input.analysesopt.VTAeVstim
        try:
            VTAeVstim = VTAplot(VTAopt, pos_VTAeVstim, input.stimopt.Estimparams,
                                'eVstim', input.save_flag and aopt.save_VTAplot, figdir=fig_dir)
        except Exception as E:
            print(E)
    if pos_VTAoptogenx is not None:
        VTAopt = input.analysesopt.VTAOptogenx
        try:
            VTAOptogenx = VTAplot(VTAopt, pos_VTAoptogenx, input.stimopt.Ostimparams,
                                  'optogenx', input.save_flag and aopt.save_VTAplot, figdir=fig_dir)
        except Exception as E:
            print(E)

    if input.plot_flag:
        plt.show(block=False)

    return iOptogenx, succes_ratio, VTAOptogenx, VTAeVstim
