import os, shutil
import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import process
from process import _determine_sit_rest_indexes, _find_mmax_p1_idxs, _find_mmax_rms, _calc_slope_of_line, _calc_lower_limit

subject = 'sub22'  # 22, 15. Old: 13
keys = ['01', '05', '10', '15', '25', '50', '75', '90', '95', '100']

# get torque and SIT amplitude for each trial
path = os.path.join('data', 'proc')
with open(os.path.join(path, 'subjects_data.json')) as file:
    df = json.load(file)

# get twitch traces for each trial
sub_info, sub_data, sub_info_short = process._import_signals(subject)
sub_data = process._calibrate_EMG_signals(sub_info, sub_data)
sub_data = process._calibrate_loadcell_signals(sub_info, sub_data)
sub_data = process._remove_loadcell_offset_start_each_trial(sub_info, sub_data)
sub_data, max_vals_and_indexes, signals_above_threshold = process._find_MVC_normalize_torque_signals(sub_info, sub_data)
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

# ----------------------------------------
# plot figure: panels are 0-indexed
# ----------------------------------------
fig = plt.figure(figsize=(11, 7))
voffset_torque, voffset_emg = 0, 0 # set increasing vertical offset as activation increases
sepby_torque, sepby_emg = 6, 1.0 # separated by this interval in %MVC or mV

gs = GridSpec(1, 3)
gs.update(left=0.1, right=0.38, bottom=0.52, wspace=0)
ax0_SO = fig.add_subplot(gs[0, 0])     # EMG SO
ax0_MG = fig.add_subplot(gs[0, 1])     # EMG MG
ax0_LG = fig.add_subplot(gs[0, 2])     # EMG LG

ax1 = fig.add_subplot(2, 9, (4,6))     # REST, SIT twitch amplitudes

ax2 = fig.add_subplot(2, 9, (7, 9)) # M waves

ax3 = fig.add_subplot(2, 9, (10,12))   # activation on torque
ax4 = fig.add_subplot(2, 9, (13,15))   # EMG on activation
ax5 = fig.add_subplot(2, 9, (16,18))   # ln(EMG) on activation

# scale bars
ax0_SO.plot([-0.002, -0.002], [0 - (sepby_emg * len(keys) - sepby_emg), 0.5 - (sepby_emg * len(keys) - sepby_emg)], 'k', linewidth=1)  # vertical
ax0_SO.text(-0.01, 0 - (sepby_emg * len(keys) - sepby_emg), '0.5 mV', rotation=90)

ax1.plot([0.27, 0.27], [-80, -70], 'k', linewidth=1)  # vertical
ax1.text(0.25, -80, '10% MVC', rotation=90)
ax1.plot([0.275, 0.05 + 0.275], [-82, -82], 'k', linewidth=1)  # horizontal
ax1.text(0.275, -88, '0.05 s')

ax2.plot([-0.2 + 900, -0.2 + 900], [-4, -4 + 0.5], 'k', linewidth=1)  # vertical
ax2.text(-1.2 + 700, -4, '0.5 mV', rotation=90)

# subplot titles
ax0_SO.text(0.025, 0.4, 'SO', horizontalalignment='center')
ax0_MG.text(0.025, 0.4, 'MG', horizontalalignment='center')
ax0_LG.text(0.025, 0.4, 'LG', horizontalalignment='center')

ax1.text(0.06, 35, 'Superimposed\ntwitches', horizontalalignment='center')
ax1.text(0.28, 35, 'Resting\ntwitches', horizontalalignment='center')

ax2.text(250, 5.5, 'SO', horizontalalignment='center')
ax2.text(1300, 5.5, 'MG', horizontalalignment='center')
ax2.text(2500, 5.5, 'LG', horizontalalignment='center')

# ----------------------------------------
# ax0, ax1 plots
# ----------------------------------------
for key, color in zip(keys, colors):
    # ----------------------------------------
    # ax0: plot rectified EMG over 50 ms
    # ----------------------------------------
    nsamples_before_trig = int(sub_info.freq * 0.05)  # mean EMG over 50 ms window
    index_rest, index_sit = _determine_sit_rest_indexes(sub_info, sub_data, key)
    # shift indexed EMG region away from filter artefact close to stimulus artefact
    filter_artefact_length = int(sub_info.freq * 0.05)
    index_start, index_stop = index_sit - (filter_artefact_length + nsamples_before_trig), index_sit - filter_artefact_length

    # EMG SO
    emg = sub_data[key].sig['emgSO'].rect[index_start: index_stop]
    emg = emg - voffset_emg
    time = sub_data[key].sig['emgSO'].times[index_start: index_stop]
    time = time - time[0]
    ax0_SO.plot(time, emg, color)
    ax0_SO.set_xticks([])
    ax0_SO.set_xticklabels([])
    ax0_SO.set_yticks([])
    ax0_SO.set_yticklabels([])
    ax0_SO.spines['top'].set_visible(False)
    ax0_SO.spines['bottom'].set_visible(False)
    ax0_SO.spines['left'].set_visible(False)
    ax0_SO.spines['right'].set_visible(False)
    ax0_SO.text(-1.45, 1.1, 'A', fontsize=14, transform=ax1.transAxes, weight='bold')

    # EMG MG
    emg = sub_data[key].sig['emgMG'].rect[index_start: index_stop]
    emg = emg - voffset_emg
    time = sub_data[key].sig['emgMG'].times[index_start: index_stop]
    time = time - time[0]
    ax0_MG.plot(time, emg, color)
    ax0_MG.set_xticks([])
    ax0_MG.set_xticklabels([])
    ax0_MG.set_yticks([])
    ax0_MG.set_yticklabels([])
    ax0_MG.spines['top'].set_visible(False)
    ax0_MG.spines['bottom'].set_visible(False)
    ax0_MG.spines['left'].set_visible(False)
    ax0_MG.spines['right'].set_visible(False)

    # EMG LG
    emg = sub_data[key].sig['emgLG'].rect[index_start: index_stop]
    emg = emg - voffset_emg
    time = sub_data[key].sig['emgLG'].times[index_start: index_stop]
    time = time - time[0]
    ax0_LG.plot(time, emg, color)
    ax0_LG.set_xticks([])
    ax0_LG.set_xticklabels([])
    ax0_LG.set_yticks([])
    ax0_LG.set_yticklabels([])
    ax0_LG.spines['top'].set_visible(False)
    ax0_LG.spines['bottom'].set_visible(False)
    ax0_LG.spines['left'].set_visible(False)
    ax0_LG.spines['right'].set_visible(False)

    # ----------------------------------------
    # ax1: plot twitch amplitudes
    # ----------------------------------------
    # find REST and SIT indexes
    index_rest, index_sit = _determine_sit_rest_indexes(sub_info, sub_data, key)
    # index 150 ms region after the REST, SIT indexes to get force
    def get_indices(index):
        index = index
        high_force_trials = ['90', '95', '100']
        if key in high_force_trials and type == 'sit':
            nsamples = int(sub_info.freq * 0.150)  # Finds twitch peak at high force within 150 ms
        else:
            nsamples = int(sub_info.freq * 0.150)  # Finds twitch peak at low-mod force within 150 ms
        index1, index2 = index, index + nsamples
        return index1, index2
    rest_index1, rest_index2 = get_indices(index_rest)
    sit_index1, sit_index2 = get_indices(index_sit)

    raw = sub_data[key].sig['torque'].raw  # torque in V
    time = sub_data[key].sig['torque'].times
    loadcell_offset_value = np.mean(sub_data['baseline'].sig['torque'].raw) * sub_info.scale_MVC_loadcell
    raw = raw * sub_info.scale_MVC_loadcell - loadcell_offset_value  # torque in Nm
    raw = raw / max_vals_and_indexes.mvc_torque[0] * 100  # torque normalised to MVC
    raw = raw - np.mean(raw[:2000])  # unfiltered
    # SIT traces
    time_sit = time[sit_index1 - sit_index1: sit_index2 - sit_index1]  # set all SIT twitch times to zero
    raw_sit = raw[sit_index1: sit_index2]
    raw_sit = raw_sit - np.mean(raw_sit[:5])
    raw_sit = raw_sit - voffset_torque
    ax1.plot(time_sit, raw_sit, color, label=key)
    if key == '01':
        ax1.text(-0.005, 0 - (voffset_torque + 1), '1%', horizontalalignment='right', size=8)
    elif key == '05':
        ax1.text(-0.005, 0 - (voffset_torque + 1), '5%', horizontalalignment='right', size=8)
    else:
        ax1.text(-0.005, 0 - (voffset_torque + 1), key + '%', horizontalalignment='right', size=8)

    # REST traces
    time_rest = time[rest_index1 - rest_index1 + int(sub_info.freq * 0.2): rest_index2 - rest_index1 + int(sub_info.freq * 0.2)]  # set all REST twitch times to 200 ms
    raw_rest = raw[rest_index1: rest_index2]
    raw_rest = raw_rest - np.mean(raw_rest[:5])
    raw_rest = raw_rest - voffset_torque
    ax1.plot(time_rest, raw_rest, color, label=key)
    ax1.text(-0.2, 1.1, 'B', fontsize=14, transform=ax1.transAxes, weight='bold')

    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    voffset_torque += sepby_torque
    voffset_emg += sepby_emg

# ----------------------------------------
# ax2: plot SIT amplitude on torque
# ----------------------------------------
for muscle in ['emgSO', 'emgMG', 'emgLG']:
    # find M wave phase 1
    idx1, idx2 = _find_mmax_p1_idxs(sub_info, sub_data)
    mmax_p1_rms, mmax_p1_avrect, mmax_new, idx_start_p1_mmax, idx_stop_p1_mmax = \
        _find_mmax_rms(sub_info, sub_data, idx1, idx2, muscle, plot_fig=False)
    samples = np.arange(0, len(mmax_new), 1)

    # offset samples (x values) to the right for MG, LG
    # interpolation was up-sampled by 10x (interpolated using step 0.1)
    offset = 0
    if muscle == 'emgMG':
        offset = 1200
    elif muscle == 'emgLG':
        offset = 2400
    samples = samples + offset
    print(muscle, idx_start_p1_mmax, idx_stop_p1_mmax, samples)

    # find limits (top, bottom, left, right) of shaded region
    slope, intercept = _calc_slope_of_line(samples[idx_start_p1_mmax], mmax_new[idx_start_p1_mmax],
                                           samples[idx_stop_p1_mmax], mmax_new[idx_stop_p1_mmax])
    lower_limit = _calc_lower_limit(samples, slope, intercept)

    # plot M waves for each muscle
    ax2.plot(samples, mmax_new, 'k')
    ax2.fill_between(samples, lower_limit, mmax_new,
                     where=(samples > samples[idx_start_p1_mmax]) & (samples < samples[idx_stop_p1_mmax]),
                     color='k', alpha=0.5, interpolate=True)

ax2.set_xticks([])
ax2.set_xticklabels([])
ax2.set_yticks([])
ax2.set_yticklabels([])
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.text(1.1, 1.1, 'C', fontsize=14, transform=ax1.transAxes, weight='bold')

# ----------------------------------------
# ax3: plot activation torque curve
# ----------------------------------------
torques, activations, emgSO_mvc, emgMG_mvc, emgLG_mvc, emgSO_mmax, emgMG_mmax, emgLG_mmax = [[] for i in range(8)]
for key in keys:
    activation = df[subject]['activations'][key]['activation']
    emgSO_mvc_trial = df[subject]['norm_emgs']['emgSO']['emg_norm_mvc'][key]['norm_mvc']
    emgMG_mvc_trial = df[subject]['norm_emgs']['emgMG']['emg_norm_mvc'][key]['norm_mvc']
    emgLG_mvc_trial = df[subject]['norm_emgs']['emgLG']['emg_norm_mvc'][key]['norm_mvc']
    emgSO_mmax_trial = df[subject]['norm_emgs']['emgSO']['emg_norm_mmax'][key]['norm_mmax']
    emgMG_mmax_trial = df[subject]['norm_emgs']['emgMG']['emg_norm_mmax'][key]['norm_mmax']
    emgLG_mmax_trial = df[subject]['norm_emgs']['emgLG']['emg_norm_mmax'][key]['norm_mmax']
    activations.append(activation)
    emgSO_mvc.append(emgSO_mvc_trial)
    emgMG_mvc.append(emgMG_mvc_trial)
    emgLG_mvc.append(emgLG_mvc_trial)
    emgSO_mmax.append(emgSO_mmax_trial)
    emgMG_mmax.append(emgMG_mmax_trial)
    emgLG_mmax.append(emgLG_mmax_trial)

ax3.plot(activations, emgSO_mvc, 'k-', label='SO')
ax3.plot(activations, emgMG_mvc, 'k--', label='MG')
ax3.plot(activations, emgLG_mvc, 'k:', label='LG')
ax3.plot(activations, emgSO_mvc, 'ko', markersize=4)
ax3.plot(activations, emgMG_mvc, 'ko', markersize=4)
ax3.plot(activations, emgLG_mvc, 'ko', markersize=4)
ax3.set_xlim(0, 100)
ax3.legend()
ax3.set_ylabel('EMG (%max MVC)')
ax3.set_xlabel('Voluntary activation (%)')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.text(-1.45, -0.15, 'D', fontsize=14, transform=ax1.transAxes, weight='bold')

# ----------------------------------------
# ax4: plot EMG activation curves
# ----------------------------------------
ax4.plot(activations, emgSO_mmax, 'k-', label='SO')
ax4.plot(activations, emgMG_mmax, 'k--', label='MG')
ax4.plot(activations, emgLG_mmax, 'k:', label='LG')
ax4.plot(activations, emgSO_mmax, 'ko', markersize=4)
ax4.plot(activations, emgMG_mmax, 'ko', markersize=4)
ax4.plot(activations, emgLG_mmax, 'ko', markersize=4)
ax4.set_xlim(0, 100)
ax4.set_ylabel(r'EMG (%M$_{\max}$)')
ax4.set_xlabel('Voluntary activation (%)')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.text(-0.2, -0.15, 'E', fontsize=14, transform=ax1.transAxes, weight='bold')

# ----------------------------------------
# ax5: plot log EMG activation curves
# ----------------------------------------
ax5.plot(emgSO_mmax, emgSO_mvc, 'k-', label='SO')
ax5.plot(emgMG_mmax, emgMG_mvc, 'k--', label='MG')
ax5.plot(emgLG_mmax, emgLG_mvc, 'k:', label='LG')
ax5.plot(emgSO_mmax, emgSO_mvc, 'ko', markersize=4)
ax5.plot(emgMG_mmax, emgMG_mvc, 'ko', markersize=4)
ax5.plot(emgLG_mmax, emgLG_mvc, 'ko', markersize=4)
ax5.set_ylabel('EMG (%max MVC)')
ax5.set_xlabel(r'EMG (%M$_{\max}$)')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.text(1.1, -0.15, 'F', fontsize=14, transform=ax1.transAxes, weight='bold')

plt.tight_layout()
plt.savefig('fig-1' + '.png', dpi=300)
shutil.move('fig-1' + '.png', os.path.join('.', 'data', 'proc', sub_info.sub, 'fig-1' + '.png'))
plt.savefig('fig-1' + '.svg', dpi=300)
shutil.move('fig-1' + '.svg', os.path.join('.', 'data', 'proc', sub_info.sub, 'fig-1' + '.svg'))
plt.close()
