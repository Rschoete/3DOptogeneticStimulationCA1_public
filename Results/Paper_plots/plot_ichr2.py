import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from neuron import h

from Functions.globalFunctions.ExtracellularField.stimulation_EcF import \
    singleSquarePulse

font = {'family': 'helvetica',
        'size': 10}

save_figs = True

mpl.rc('font', **font)
h.nrn_load_dll("./Model/Mods/nrnmech.dll")
h.load_file('stdrun.hoc')

sns.color_palette("crest", as_cmap=True)

soma = h.Section('soma')
loc = soma
loc.insert('chr2h134r')
loc.insert('xtra')
vc = h.VClamp(soma(0.5))


for seg in loc:  # connectpoint to connect mod-files chr2h13r en optopulse
    # Intensity Iopto refers to intensity profile as result of point process h.OPTO_pulse
    seg.os_xtra = 1
    h.setpointer(seg.xtra._ref_es, 'ex', seg.xtra)
    h.setpointer(seg.chr2h134r._ref_Iopto, 'ox', seg.xtra)

# Change gchr2bar to make the cell more sensitive
loc(0.5).chr2h134r.gchr2bar = 50  # default: 0.4 mS/cm2


# run stimulation
osstart = 10
osdur = 100
osAmp = 1000  # gchr2bar aanpassen om AP te verkrijgen: zie verschillende waarden
ostim_time, ostim_amp = singleSquarePulse(delay=osstart, dur=osdur, amp=osAmp)
ostim_amp = h.Vector(ostim_amp)
ostim_time = h.Vector(ostim_time)
ostim_amp.play(h._ref_ostim_xtra, ostim_time,
               True)

vc.dur[0] = osstart
vc.dur[1] = osdur+300
vc.dur[2] = 100
vc.amp[0] = -70
vc.amp[1] = 0
vc.amp[2] = -70

ichr2_list = []
vs_list = []
t_list = []
ostims = []
ostim_time_list = []

# record
Dt = 0.1  # ms tijdsstap voor opslaan; arbitrair
t = h.Vector().record(h._ref_t, Dt)
v_s = h.Vector().record(loc(0.5)._ref_v, Dt)
ichr2 = h.Vector().record(loc(0.5)._ref_i_chr2h134r, Dt)

osdurs = [1, 100]
Iamps = np.logspace(0, 4, 5)
for Iamp in Iamps:
    vs_list_intm = []
    t_list_intm = []
    ichr2_list_intm = []
    ostim_list_intm = []
    ostim_time_list_intm = []
    for osdur in osdurs:
        vc.dur[0] = osstart
        vc.dur[1] = osdur+300
        vc.dur[2] = 100
        vc.amp[0] = -70
        vc.amp[1] = -70
        vc.amp[2] = -70
        ostim_time, ostim_amp = singleSquarePulse(
            delay=osstart, dur=osdur, amp=Iamp)
        ostim_amp = h.Vector(ostim_amp)
        ostim_time = h.Vector(ostim_time)
        ostim_amp.play(h._ref_ostim_xtra, ostim_time,
                       True)

        h.finitialize(-70)
        h.continuerun(500)

        vs_list_intm.append(np.array(v_s))
        ichr2_list_intm.append(np.array(ichr2))
        t_list_intm.append(np.array(t))
        ostim_list_intm.append(np.array(ostim_amp))
        ostim_time_list_intm.append(np.array(ostim_time))

    vs_list.append(vs_list_intm)
    ichr2_list.append(ichr2_list_intm)
    t_list.append(t_list_intm)
    ostims.append(ostim_list_intm)
    ostim_time_list.append(ostim_time_list_intm)


#colors = np.array(sns.color_palette("crest", n_colors=len(Iamps)))
colors = np.array(sns.light_palette("seagreen", n_colors=len(Iamps)))
# plot membrane voltage for all sections
fig = plt.figure(tight_layout=True, figsize=(8/2.54, 5/2.54))
gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[1, 2])

axs = []
axs.append(fig.add_subplot(gs[0]))
axs.append(fig.add_subplot(gs[1]))
for it, ivs, iichr2, oamp, ostime, clr in zip(t_list, vs_list, ichr2_list, ostims, ostim_time_list, colors):
    for i, osdur in enumerate(osdurs):

        if len(oamp[i] > 0):
            Iopt_np = np.array(oamp[i])
            idx_on = (Iopt_np > 0) & (np.roll(Iopt_np, 1) <= 0)
            idx_off = (Iopt_np > 0) & (np.roll(Iopt_np, -1) <= 0)
            t_np = np.array(ostime[i])
            t_on = t_np[idx_on]
            t_off = t_np[idx_off]
            if len(t_on) > len(t_off):
                # if illumination till end of simulaton time t_off could miss a final time point
                t_off = np.append(t_off, t_np[-1])
            for ton, toff in zip(t_on, t_off):
                axs[i].axvspan(ton, toff, color='silver', alpha=0.2)
        axs[i].plot(it[i], iichr2[i], color=clr, alpha=1)


axs[0].set_xlim([0, 100])
axs[1].set_xlim([0, 200])
axs[0].set_ylim([-0.18, 0.01])
axs[1].set_ylim([-0.18, 0.01])
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axs[1].spines['left'].set_visible(False)
axs[1].set_yticks([])
axs[0].set_ylabel('$i_{chr2}$ [$\mu$A/$cm^2$]')
if save_figs:
    fig.savefig('opsincurrent.svg')


ichr2_list = []
ichr2_peak_list = []
ichr2_ss_list = []
vs_list = []
t_list = []
ostims = []
ostim_time_list = []

# record
Dt = 0.1  # ms tijdsstap voor opslaan; arbitrair
t = h.Vector().record(h._ref_t, Dt)
v_s = h.Vector().record(loc(0.5)._ref_v, Dt)
ichr2 = h.Vector().record(loc(0.5)._ref_i_chr2h134r, Dt)

osdurs = [1, 100]
Iamps = np.logspace(-2, 6, 17)
for Iamp in Iamps:
    vs_list_intm = []
    t_list_intm = []
    ichr2_list_intm = []
    ostim_list_intm = []
    ostim_time_list_intm = []
    ichr2_peak_list_intm = []
    ichr2_ss_list_intm = []
    for osdur in osdurs:
        vc.dur[0] = osstart
        vc.dur[1] = osdur+300
        vc.dur[2] = 100
        vc.amp[0] = -70
        vc.amp[1] = -70
        vc.amp[2] = -70
        ostim_time, ostim_amp = singleSquarePulse(
            delay=osstart, dur=osdur, amp=Iamp)
        ostim_amp = h.Vector(ostim_amp)
        ostim_time = h.Vector(ostim_time)
        ostim_amp.play(h._ref_ostim_xtra, ostim_time,
                       True)

        h.finitialize(-70)
        h.continuerun(500)

        ichr2_tmp = np.array(ichr2)
        t_tmp = np.array(t)
        ipeak = np.min(ichr2_tmp)
        iss = ichr2_tmp[np.argmin(np.abs(t_tmp-(osstart+osdur)))]
        vs_list_intm.append(np.array(v_s))
        ichr2_list_intm.append(np.array(ichr2))
        t_list_intm.append(np.array(t))
        ostim_list_intm.append(np.array(ostim_amp))
        ostim_time_list_intm.append(np.array(ostim_time))
        ichr2_peak_list_intm.append(ipeak)
        ichr2_ss_list_intm.append(iss)

    vs_list.append(vs_list_intm)
    ichr2_list.append(ichr2_list_intm)
    t_list.append(t_list_intm)
    ostims.append(ostim_list_intm)
    ostim_time_list.append(ostim_time_list_intm)
    ichr2_peak_list.append(ichr2_peak_list_intm)
    ichr2_ss_list.append(ichr2_ss_list_intm)

fig2 = plt.figure(tight_layout=True, figsize=(8/2.54, 3.8/2.54))
ax = fig2.add_subplot(111)
ichr2_peak_list = np.array(ichr2_peak_list)
ichr2_ss_list = np.array(ichr2_ss_list)

ax.plot(Iamps/1000, -ichr2_peak_list[:, 0], color='seagreen')
ax.plot(Iamps/1000, -ichr2_ss_list[:, 0], color='lightgreen', linestyle='--')
ax.plot(Iamps/1000, -ichr2_peak_list[:, 1], color='steelblue')
ax.plot(Iamps/1000, -ichr2_ss_list[:, 1], color='lightblue', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('$i_{chr2}$ [$\mu$A/$cm^2$]')
ax.set_xlabel('Irr [mW/$mm^2$]')
ax.set_yticks([1e-5, 1e-3, 1e-1])
if save_figs:
    fig2.savefig('peak_sscurrent.svg')
plt.show(block='on')
