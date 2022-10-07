import os, shutil
import numpy as np
import scipy.signal
from scipy import interpolate, stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
import pandas as pd
from collections import namedtuple
import utilities as utils
import trials_key
import spike2py as spk2
from scipy.signal import butter, filtfilt

LOCAL = '/home/joanna/Dropbox/Projects/methodsEMG' # path for processed data
DATA = '/media/joanna/Elements/Projects/activation' # path for raw data
REPO = '.'
os.chdir(REPO); print(os.getcwd())
plot_fig = False # set for hidden functions


def import_process_signals(subjects):
    subjects_data = dict()
    for subject in subjects:
        _mkdir_proc(subject)
        sub_info, sub_data, sub_info_short = _import_signals(subject)
        sub_data = _calibrate_EMG_signals(sub_info, sub_data)
        sub_data = _calibrate_loadcell_signals(sub_info, sub_data)
        sub_data = _remove_loadcell_offset_start_each_trial(sub_info, sub_data)
        sub_data, max_vals_and_indexes, signals_above_threshold = _find_MVC_normalize_torque_signals(sub_info, sub_data)
        activations = _calculate_activations(sub_info, sub_data, max_vals_and_indexes)
        torques_emgs = _calculate_torque_EMG_at_activations(sub_info, sub_data)

        # EMG normalisation: to MVC and MMax
        norm_emgs = dict()
        mvc_emg_torque_times = dict()
        emgs_rect, emgs_rms = _find_trial_emg_rms(sub_info, sub_data, plot_fig=False)
        for muscle in ['emgSO', 'emgMG', 'emgLG']:
            idx1, idx2 = _find_mmax_p1_idxs(sub_info, sub_data)
            mmax_p1_rms, mmax_p1_avrect, mmax_new, idx_start_p1_mmax, idx_stop_p1_mmax = _find_mmax_rms(sub_info, sub_data, idx1, idx2, muscle=muscle, plot_fig=True) # plots mmax-p1-emg.png
            if subject in ['sub01', 'sub02', 'sub03']:
                print('sub: {}, muscle: {}, mmax rms: {:.4f}, mmax avrect: {:.4f}'.format(subject, muscle, mmax_p1_rms, mmax_p1_avrect))
            mvc_emg_rms = _find_mvc_emg_rms(sub_info, max_vals_and_indexes, signals_above_threshold, muscle=muscle, plot_fig=True) # plots mvc_rms_emg.png
            emg_norm_mvc, emg_norm_mmax = _normalise_emg(sub_data, mvc_emg_rms, mmax_p1_rms, emgs_rms, muscle=muscle)
            diff_maxRect_torque, diff_mvwin_torque = _find_time_difference_btw_mvcEMG_mvcTorque(sub_info, max_vals_and_indexes, signals_above_threshold, muscle, plot_fig=True)
            norm_emgs.update({muscle: {'emg_norm_mvc': emg_norm_mvc,
                                       'emg_norm_mmax': emg_norm_mmax}})
            mvc_emg_torque_times.update({muscle: {'diff_maxRect_torque': diff_maxRect_torque,
                                                  'diff_mvwin_torque': diff_mvwin_torque}})

        subjects_data.update({subject: {'sub_info': sub_info_short,
                                        'mvc_torque': max_vals_and_indexes.mvc_torque[0],
                                        'activations': activations, # Access activations and torques with trial key
                                        'torques_emgs': torques_emgs,
                                        'norm_emgs': norm_emgs,
                                        'mvc_emg_torque_times': mvc_emg_torque_times}})
    return subjects_data

def write_signals_to_json(subjects_data):
    path = os.path.join('.', 'data', 'proc')
    with open(os.path.join(path, 'subjects_data.json'), 'w') as file:
        json.dump(subjects_data, file)

def write_sub_info_to_csv(subjects, subjects_data):
    path = os.path.join('.', 'data', 'proc')
    df = pd.DataFrame()
    for subject in subjects:
        age, sex, height, weight, act_base, mvc_torque = [[] for i in range(6)]
        age.append(subjects_data[subject]['sub_info']['age'])
        sex.append(subjects_data[subject]['sub_info']['sex'])
        height.append(subjects_data[subject]['sub_info']['height'])
        weight.append(subjects_data[subject]['sub_info']['weight'])
        activation = np.array(subjects_data[subject]['sub_info']['activations_baseline']).mean() # mean of 2 activation MVCs
        act_base.append(activation)
        mvc_torque.append(subjects_data[subject]['mvc_torque'])
        df_ = pd.DataFrame({'subject': subject, 'age': age, 'sex': sex, 'height': height, 'weight': weight,
                            'act_base': act_base, 'mvc_torque': mvc_torque})
        df = df.append(df_, ignore_index=False)
    df.to_csv(os.path.join(path,'subjects_info.csv'))
    df.describe().to_csv(os.path.join(path, 'subjects_describe.csv'))

def write_signals_to_csv(subjects, subjects_data):
    path = os.path.join('.', 'data', 'proc')
    keys = ['01', '05', '10', '15', '25', '50', '75', '90', '95', '100']
    df = pd.DataFrame()
    for subject in subjects:
        sub = list((subject,) * len(keys))
        activations, torques, emgSO, emgMG, emgLG, \
        emgSO_norm_mvc, emgSO_norm_mmax, \
        emgMG_norm_mvc, emgMG_norm_mmax, \
        emgLG_norm_mvc, emgLG_norm_mmax = [[] for i in range(11)]
        for key in keys:
            activations.append(subjects_data[subject]['activations'][key]['activation'])
            torques.append(subjects_data[subject]['torques_emgs'][key]['torque'])
            emgSO.append(subjects_data[subject]['torques_emgs'][key]['emgSO'])
            emgMG.append(subjects_data[subject]['torques_emgs'][key]['emgMG'])
            emgLG.append(subjects_data[subject]['torques_emgs'][key]['emgLG'])
            emgSO_norm_mvc.append(subjects_data[subject]['norm_emgs']['emgSO']['emg_norm_mvc'][key]['norm_mvc'])
            emgSO_norm_mmax.append(subjects_data[subject]['norm_emgs']['emgSO']['emg_norm_mmax'][key]['norm_mmax'])
            emgMG_norm_mvc.append(subjects_data[subject]['norm_emgs']['emgMG']['emg_norm_mvc'][key]['norm_mvc'])
            emgMG_norm_mmax.append(subjects_data[subject]['norm_emgs']['emgMG']['emg_norm_mmax'][key]['norm_mmax'])
            emgLG_norm_mvc.append(subjects_data[subject]['norm_emgs']['emgLG']['emg_norm_mvc'][key]['norm_mvc'])
            emgLG_norm_mmax.append(subjects_data[subject]['norm_emgs']['emgLG']['emg_norm_mmax'][key]['norm_mmax'])
        df_ = pd.DataFrame({'subject': sub, 'trials': keys, 'activations': activations, 'torques': torques,
                            'emgSO': emgSO, 'emgMG': emgMG, 'emgLG': emgLG,
                            'lnemgSO': np.log(emgSO), 'lnemgMG': np.log(emgMG), 'lnemgLG': np.log(emgLG),
                            'emgSO_norm_mvc': emgSO_norm_mvc, 'emgSO_norm_mmax': emgSO_norm_mmax,
                            'emgMG_norm_mvc': emgMG_norm_mvc, 'emgMG_norm_mmax': emgMG_norm_mmax,
                            'emgLG_norm_mvc': emgLG_norm_mvc, 'emgLG_norm_mmax': emgLG_norm_mmax})
        df = df.append(df_, ignore_index=False)
    df.to_csv(os.path.join(path,'subjects_data.csv'))

def write_times_to_csv(subjects):
    path = os.path.join('.', 'data', 'proc')
    with open(os.path.join(path, 'subjects_data.json'), 'r') as file:
        data = json.load(file)

    diff_maxRect_torque_emgSO, diff_maxRect_torque_emgMG, diff_maxRect_torque_emgLG, \
    diff_mvwin_torque_emgSO,diff_mvwin_torque_emgMG, diff_mvwin_torque_emgLG = [[] for i in range(6)]
    for subject in subjects:
        diff_maxRect_torque_emgSO.append(data[subject]['mvc_emg_torque_times']['emgSO']['diff_maxRect_torque'])
        diff_maxRect_torque_emgMG.append(data[subject]['mvc_emg_torque_times']['emgMG']['diff_maxRect_torque'])
        diff_maxRect_torque_emgLG.append(data[subject]['mvc_emg_torque_times']['emgLG']['diff_maxRect_torque'])
        diff_mvwin_torque_emgSO.append(data[subject]['mvc_emg_torque_times']['emgSO']['diff_mvwin_torque'])
        diff_mvwin_torque_emgMG.append(data[subject]['mvc_emg_torque_times']['emgMG']['diff_mvwin_torque'])
        diff_mvwin_torque_emgLG.append(data[subject]['mvc_emg_torque_times']['emgLG']['diff_mvwin_torque'])

    df = pd.DataFrame({'subject': subjects,
                       'diff_maxRect_torque_emgSO': diff_maxRect_torque_emgSO,
                       'diff_maxRect_torque_emgMG': diff_maxRect_torque_emgMG,
                       'diff_maxRect_torque_emgLG': diff_maxRect_torque_emgLG,
                       'diff_mvwin_torque_emgSO': diff_mvwin_torque_emgSO,
                       'diff_mvwin_torque_emgMG': diff_mvwin_torque_emgMG,
                       'diff_mvwin_torque_emgLG': diff_mvwin_torque_emgLG})
    df.to_csv(os.path.join(path,'subjects_times_mvc.csv'))

def plot_emg_activation(subjects, plot_fig=True):
    path = os.path.join('.', 'data', 'proc')
    with open(os.path.join(path, 'subjects_data.json'), 'r') as file:
        data = json.load(file)
    keys = ['01', '05', '10', '15', '25', '50', '75', '90', '95', '100']

    if plot_fig:
        # plot EMG-activation
        fig = plt.figure(figsize=(9, 11))
        ax1 = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        ax3 = fig.add_subplot(3, 2, 3)
        ax4 = fig.add_subplot(3, 2, 4)
        ax5 = fig.add_subplot(3, 2, 5)
        ax6 = fig.add_subplot(3, 2, 6)
        for subject in subjects:
            activation, emgSO, emgMG, emgLG = [[] for i in range(4)]
            for key in keys:
                activation.append(data[subject]['activations'][key]['activation'])
                emgSO.append(data[subject]['torques_emgs'][key]['emgSO'])
                emgMG.append(data[subject]['torques_emgs'][key]['emgMG'])
                emgLG.append(data[subject]['torques_emgs'][key]['emgLG'])
            # log the EMG signals
            emgSO_ln = np.log(emgSO)
            emgMG_ln = np.log(emgMG)
            emgLG_ln = np.log(emgLG)
            # plot
            ax1.plot(activation, emgSO)
            ax2.plot(activation, emgSO_ln)
            ax3.plot(activation, emgMG)
            ax4.plot(activation, emgMG_ln)
            ax5.plot(activation, emgLG)
            ax6.plot(activation, emgLG_ln)

        ax1.set_ylabel('EMG SO [mV]')
        ax2.set_ylabel('EMG SO [ln(mV)]')
        ax3.set_ylabel('EMG MG [mV]')
        ax4.set_ylabel('EMG MG [ln(mV)]')
        ax5.set_ylabel('EMG LG [mV]')
        ax5.set_xlabel('Activation [%]')
        ax6.set_ylabel('EMG LG [ln(mV)]')
        ax6.set_xlabel('Activation [%]')
        ax1.autoscale(enable=True, axis='x', tight=True)
        ax2.autoscale(enable=True, axis='x', tight=True)
        ax3.autoscale(enable=True, axis='x', tight=True)
        ax4.autoscale(enable=True, axis='x', tight=True)
        ax5.autoscale(enable=True, axis='x', tight=True)
        ax6.autoscale(enable=True, axis='x', tight=True)
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'emg_activation.png'), dpi=300)
        plt.close()

def plot_activation_torque(subjects, plot_fig=True):
    path = os.path.join('.', 'data', 'proc')
    with open(os.path.join(path, 'subjects_data.json'), 'r') as file:
        data = json.load(file)
    keys = ['01', '05', '10', '15', '25', '50', '75', '90', '95', '100']

    if plot_fig:
        # plot activation-torque
        fig = plt.figure(figsize=(11, 7))
        for subject in subjects:
            activation, torque = [[] for i in range(2)]
            for key in keys:
                activation.append(data[subject]['activations'][key]['activation'])
                torque.append(data[subject]['torques_emgs'][key]['torque'])
            plt.plot(torque, activation, '0.5')
        plt.ylabel('Activation [%]')
        plt.xlabel('Torque [%MVC]')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'activation_torque.png'), dpi=300)
        plt.close()

def plot_emg_torque(subjects, plot_fig=True):
    path = os.path.join('.', 'data', 'proc')
    with open(os.path.join(path, 'subjects_data.json'), 'r') as file:
        data = json.load(file)
    keys = ['01', '05', '10', '15', '25', '50', '75', '90', '95', '100']

    if plot_fig:
        # plot emg-torque
        fig = plt.figure(figsize=(4, 6))
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)
        for subject in subjects:
            torque, emgSO, emgMG, emgLG = [[] for i in range(4)]
            for key in keys:
                torque.append(data[subject]['torques_emgs'][key]['torque'])
                emgSO.append(data[subject]['torques_emgs'][key]['emgSO'])
                emgMG.append(data[subject]['torques_emgs'][key]['emgMG'])
                emgLG.append(data[subject]['torques_emgs'][key]['emgLG'])
            # plot
            ax1.plot(torque, emgSO, 'k', alpha=0.5)
            ax2.plot(torque, emgMG, 'k', alpha=0.5)
            ax3.plot(torque, emgLG, 'k', alpha=0.5)

        ax1.set_ylabel('EMG SO (mV)')
        ax1.set_xlabel('Torque (%MVC)')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.locator_params(axis='y', nbins=5)
        ax1.autoscale(enable=True, axis='x', tight=True)

        ax2.set_ylabel('EMG MG (mV)')
        ax2.set_xlabel('Torque (%MVC)')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.locator_params(axis='y', nbins=5)
        ax2.autoscale(enable=True, axis='x', tight=True)

        ax3.set_ylabel('EMG LG (mV)')
        ax3.set_xlabel('Torque (%MVC)')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.locator_params(axis='y', nbins=5)
        ax3.autoscale(enable=True, axis='x', tight=True)

        plt.tight_layout()
        plt.savefig(os.path.join(path, 'emg_torque.png'), dpi=300)
        plt.savefig(os.path.join(path, 'emg_torque.svg'), dpi=300)
        plt.close()

def plot_normalised_emg_calc_slopes(subjects, plot_fig=True):
    path = os.path.join('.', 'data', 'proc')
    with open(os.path.join(path, 'subjects_data.json'), 'r') as file:
        data = json.load(file)
    keys = ['01', '05', '10', '15', '25', '50', '75', '90', '95', '100']

    if plot_fig:
        # plot figure
        fig = plt.figure(figsize=(9, 9))
        ax1 = fig.add_subplot(3, 3, 1)
        ax2 = fig.add_subplot(3, 3, 2)
        ax3 = fig.add_subplot(3, 3, 3)
        ax4 = fig.add_subplot(3, 3, 4)
        ax5 = fig.add_subplot(3, 3, 5)
        ax6 = fig.add_subplot(3, 3, 6)
        ax7 = fig.add_subplot(3, 3, 7)
        ax8 = fig.add_subplot(3, 3, 8)
        ax9 = fig.add_subplot(3, 3, 9)

    slopes_all_muscles = {}
    for index, muscle in enumerate(['SO', 'MG', 'LG']):
        slopes_each_muscle = []
        for subject in subjects:
            # extract activation, EMG data from subject dict
            activations, emg_norm_mvc, emg_norm_mmax = [], [], []
            for key in keys:
                activations.append(data[subject]['activations'][key]['activation'])
                if muscle == 'SO':
                    emg_norm_mvc.append(data[subject]['norm_emgs']['emgSO']['emg_norm_mvc'][key]['norm_mvc'])
                    emg_norm_mmax.append(data[subject]['norm_emgs']['emgSO']['emg_norm_mmax'][key]['norm_mmax'])
                elif muscle == 'MG':
                    emg_norm_mvc.append(data[subject]['norm_emgs']['emgMG']['emg_norm_mvc'][key]['norm_mvc'])
                    emg_norm_mmax.append(data[subject]['norm_emgs']['emgMG']['emg_norm_mmax'][key]['norm_mmax'])
                elif muscle == 'LG':
                    emg_norm_mvc.append(data[subject]['norm_emgs']['emgLG']['emg_norm_mvc'][key]['norm_mvc'])
                    emg_norm_mmax.append(data[subject]['norm_emgs']['emgLG']['emg_norm_mmax'][key]['norm_mmax'])

            # get slopes
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(emg_norm_mmax, emg_norm_mvc)
            slopes_each_muscle.append(slope)

            # plot
            if plot_fig:
                if muscle == 'SO':
                    ax1.plot(activations, emg_norm_mvc, 'k', alpha=0.5)
                    ax2.plot(activations, emg_norm_mmax, 'k', alpha=0.5)
                    ax3.plot(emg_norm_mmax, emg_norm_mvc, 'k', alpha=0.5)
                elif muscle == 'MG':
                    ax4.plot(activations, emg_norm_mvc, 'k', alpha=0.5)
                    ax5.plot(activations, emg_norm_mmax, 'k', alpha=0.5)
                    ax6.plot(emg_norm_mmax, emg_norm_mvc, 'k', alpha=0.5)
                elif muscle == 'LG':
                    ax7.plot(activations, emg_norm_mvc, 'k', alpha=0.5)
                    ax8.plot(activations, emg_norm_mmax, 'k', alpha=0.5)
                    ax9.plot(emg_norm_mmax, emg_norm_mvc, 'k', alpha=0.5)

        if muscle == 'SO':
            d = {'slopes_SO': slopes_each_muscle}
        elif muscle == 'MG':
            d = {'slopes_MG': slopes_each_muscle}
        elif muscle == 'LG':
            d = {'slopes_LG': slopes_each_muscle}
        slopes_all_muscles.update(d)

    if plot_fig:
        ax1.set_ylabel('EMG SO (%max MVC)')
        ax1.set_xlabel('Voluntary activation (%)')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_ylim([0, 400])
        ax1.autoscale(enable=True, axis='x', tight=True)
        ax1.locator_params(axis='y', nbins=4)

        ax2.set_ylabel(r'EMG SO (%M$_{\max}$)')
        ax2.set_xlabel('Voluntary activation (%)')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_ylim([0, 50])
        ax2.autoscale(enable=True, axis='x', tight=True)

        ax3.set_ylabel('EMG SO (%max MVC)')
        ax3.set_xlabel(r'EMG SO (%M$_{\max}$)')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.set_xlim([0, 50])
        ax3.set_ylim([0, 400])
        # ax3.autoscale(enable=True, axis='x', tight=True)
        ax3.locator_params(axis='y', nbins=4)

        ax4.set_ylabel('EMG MG (%max MVC)')
        ax4.set_xlabel('Voluntary activation (%)')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.set_ylim([0, 400])
        ax4.autoscale(enable=True, axis='x', tight=True)
        ax4.locator_params(axis='y', nbins=4)

        ax5.set_ylabel(r'EMG MG (%M$_{\max}$)')
        ax5.set_xlabel('Voluntary activation (%)')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.set_ylim([0, 50])
        ax5.autoscale(enable=True, axis='x', tight=True)

        ax6.set_ylabel('EMG MG (%max MVC)')
        ax6.set_xlabel(r'EMG MG (%M$_{\max}$)')
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.set_xlim([0, 50])
        ax6.set_ylim([0, 400])
        # ax6.autoscale(enable=True, axis='x', tight=True)
        ax6.locator_params(axis='y', nbins=4)

        ax7.set_ylabel('EMG LG (%max MVC)')
        ax7.set_xlabel('Voluntary activation (%)')
        ax7.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)
        ax7.set_ylim([0, 400])
        ax7.autoscale(enable=True, axis='x', tight=True)
        ax7.locator_params(axis='y', nbins=4)

        ax8.set_ylabel(r'EMG LG (%M$_{\max}$)')
        ax8.set_xlabel('Voluntary activation (%)')
        ax8.spines['top'].set_visible(False)
        ax8.spines['right'].set_visible(False)
        ax8.set_ylim([0, 50])
        ax8.autoscale(enable=True, axis='x', tight=True)

        ax9.set_ylabel('EMG LG (%max MVC)')
        ax9.set_xlabel(r'EMG LG (%M$_{\max}$)')
        ax9.spines['top'].set_visible(False)
        ax9.spines['right'].set_visible(False)
        ax9.set_xlim([0, 50])
        ax9.set_ylim([0, 400])
        # ax9.autoscale(enable=True, axis='x', tight=True)
        ax9.locator_params(axis='y', nbins=4)

        plt.tight_layout()
        plt.savefig(os.path.join(path, 'emg_normalised.png'), dpi=300)
        plt.savefig(os.path.join(path, 'emg_normalised.svg'), dpi=300)
        plt.close()
        
    return slopes_all_muscles

def analyse_slopes_plot_hist(subjects, slopes_all_muscles, plot_fig=True):
    path = os.path.join('.', 'data', 'proc')

    file = os.path.join(path, 'results.txt')
    open(file, 'w').close()

    n = len(subjects)
    for muscle in ['slopes_SO', 'slopes_MG', 'slopes_LG']:
        slopes = slopes_all_muscles[muscle]
        slopes_log = np.log(slopes)

        mean, sd = np.mean(slopes_log), np.std(slopes_log)
        se = sd / np.sqrt(len(slopes_log))

        ci_ll, ci_ul = stats.norm.interval(0.95, loc=mean, scale=se)
        pi_ll, pi_ul = mean - 2.064 * se * np.sqrt(n), mean + 2.064 * se * np.sqrt(n) # t = 2.064 for 24 dof, from https://www.tdistributiontable.com/
        # ll, ul = mean - 1.96 * se, mean + 1.96 * se # checked -- correct

        with open(file, 'a') as f:
            f.write('\n\n{}'.format(muscle))
            f.write('\nMean {:.2f}, 95% CI {:.2f} to {:.2f}, 95% PI {:.2f} to {:.2f}'
                    .format(np.exp(mean), np.exp(ci_ll), np.exp(ci_ul), np.exp(pi_ll), np.exp(pi_ul)))

        if plot_fig:
            # plot histograms of slopes in natural units
            fig = plt.subplots(figsize=(4, 3))
            ax = plt.subplot(1, 1, 1)
            ax.hist(slopes, density=True, histtype='stepfilled', alpha=0.2)
            ax.set_xlabel('slopes')
            ax.set_ylabel('probability')
            plt.tight_layout()
            plt.savefig(os.path.join(path, muscle + '.png'), dpi=300)
            plt.close()

def write_slopes_to_csv(subjects, slopes_all_muscles):
    path = os.path.join('.', 'data', 'proc')
    with open(os.path.join(path, 'subjects_data.json'), 'r') as file:
        data = json.load(file)

    age, sex = [], []
    for subject in subjects:
        age.append(data[subject]['sub_info']['age'])
        sex.append(data[subject]['sub_info']['sex'].strip())

    df = pd.DataFrame({'subject': subjects, 'age': age, 'sex': sex,
                       'slope_SO': slopes_all_muscles['slopes_SO'],
                       'slope_MG': slopes_all_muscles['slopes_MG'],
                       'slope_LG': slopes_all_muscles['slopes_LG']})
    df['sex_coded'] = 0
    df.loc[df['sex'] == 'F', 'sex_coded'] = 1
    df.to_csv(os.path.join(path, 'subjects_slopes.csv'))

    # t-test to determine if any sex differences
    file = os.path.join(path, 'results_slope_sex.txt')
    open(file, 'w').close()
    for muscle in ['SO', 'MG', 'LG']:
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['slope_' + muscle], df.sex_coded)

        with open(file, 'a') as f:
            f.write('\n\n{}'.format(muscle))
            f.write('\nMean difference {:.2f}, p_value {:.2f}'.format(slope, p_value))

def get_mean_normalised_emg_at_activation_levels():
    path = os.path.join('.', 'data', 'proc')
    df = pd.read_csv(os.path.join(path, 'subjects_data.csv'))

    variables = ['activations', 'emgSO_norm_mvc', 'emgSO_norm_mmax', 'emgMG_norm_mvc', 'emgMG_norm_mmax', 'emgLG_norm_mvc', 'emgLG_norm_mmax']
    target_activations = [1, 5, 10, 15, 25, 50, 75, 90, 95, 100]
    df_ = pd.DataFrame({'target_activations': target_activations})

    for variable in variables:
        means, stds = [], []
        for target_activation in target_activations:
            # df['activations'][df['trials'] == 10].describe()
            mean = df[variable][df['trials'] == target_activation].describe()['mean']
            std = df[variable][df['trials'] == target_activation].describe()['std']

            means.append(mean)
            stds.append(std)

        df_[variable + '_means'] = means
        df_[variable + '_stds'] = stds

    df_.to_csv(os.path.join(path, 'subjects_data_describe.csv'))

def analyse_times(plot_fig=plot_fig):
    path = os.path.join('.', 'data', 'proc')
    df = pd.read_csv(os.path.join(path, 'subjects_times_mvc.csv'))

    if plot_fig:
        fig = plt.figure(figsize=(9, 9))
        varlist = list(df.columns.values[2:])
        axes = [1, 2, 3, 4, 5, 6]
        xlabs = ['maxRect SO (s)', 'maxRect MG (s)', 'maxRect LG (s)', 'mvwin SO (s)', 'mvwin MG (s)', 'mvwin LG (s)']
        for var, ax, xlab in zip(varlist, axes, xlabs):
            axis = fig.add_subplot(2, 3, ax)
            axis.hist(df[var], density=False, histtype='stepfilled', alpha=0.2)
            axis.set_xlabel(xlab)
            axis.set_ylabel('counts')

        plt.tight_layout()
        plt.savefig(os.path.join(path, 'diff_mvc_times.png'), dpi=300)
        plt.close()

def copy_figs(subjects):
    if not os.path.exists(os.path.join('.', 'data', 'proc', 'figs')):
        os.chdir(os.path.join('.', 'data', 'proc'))
        os.mkdir('figs')
        os.chdir(os.path.join('..', '..'))  # return to base directory: data

    for subject in subjects:
        shutil.copy(os.path.join('.', 'data', 'proc', subject, 'mmax-p1.png'),
                    os.path.join('.', 'data', 'proc', 'figs', 'mmax-p1-' + subject + '.png'))
        shutil.copy(os.path.join('.', 'data', 'proc', subject, 'mvc_torq_emg.png'),
                    os.path.join('.', 'data', 'proc', 'figs', 'mvc_torq_emg-' + subject + '.png'))
        shutil.copy(os.path.join('.', 'data', 'proc', subject, 'emg_rect.png'),
                    os.path.join('.', 'data', 'proc', 'figs', 'emg_rect-' + subject + '.png'))

def _mkdir_proc(subject):

    if not os.path.exists(os.path.join('.', 'data', 'proc', subject)):
        os.chdir(os.path.join('.', 'data', 'proc'))
        os.mkdir(subject)
        os.chdir(os.path.join('..', '..'))  # return to base directory: data

def _import_signals(subject):

    path = os.path.join(REPO, 'data', 'raw', subject)
    sub_info = utils.read_subject_log(path, subject)
    sub_key = trials_key.gen(sub_info.sub)
    signals = utils.generate_spike2py_signalInfos(sub_info)
    sub_info_short = {'age': sub_info.age, 'sex': sub_info.sex, 'height':sub_info.height, 'weight': sub_info.weight,
                      'activations_baseline': sub_info.activations_baseline}

    trial_data = dict()
    for trial, trialname in sub_key.items():
        filename = trialname + '.mat'
        trial_info = spk2.TrialInfo(cond=trial, path=path, filename=filename, signals=signals)
        trial_data[trial] = spk2.Trial(trial_info)

    return sub_info, trial_data, sub_info_short

def _calibrate_EMG_signals(sub_info, sub_data):
    """Correct EMG signals that were imported at 30% of their true amplitude"""
    fs = int(sub_info.freq)
    def envelop(signal, fs, cutoff=5):
        """Create a signal envelop for sEMG.
        Parameters
        ----------
        cutoff : int, default 5
            Cutoff frequency for lowpass filter
        """
        b, a = butter(N=4,
                      Wn=np.array(cutoff) / (fs / 2),
                      btype='lowpass',
                      )
        signal_envelop = filtfilt(b, a, np.abs(signal))
        return signal_envelop

    for key in sub_data.keys():
        # raw signal is underestimated; calibrate the processed signal
        sub_data[key].sig['emgSO'].calibrate(slope=1 / 0.3, offset=0)
        sub_data[key].sig['emgMG'].calibrate(slope=1 / 0.3, offset=0)
        sub_data[key].sig['emgLG'].calibrate(slope=1 / 0.3, offset=0)
        sub_data[key].sig['emgTA'].calibrate(slope=1 / 0.3, offset=0)

        # calculate rectified signal using the calibrated processed signal
        sub_data[key].sig['emgSO'].rect = np.abs(sub_data[key].sig['emgSO'].proc)
        sub_data[key].sig['emgMG'].rect = np.abs(sub_data[key].sig['emgMG'].proc)
        sub_data[key].sig['emgLG'].rect = np.abs(sub_data[key].sig['emgLG'].proc)
        sub_data[key].sig['emgTA'].rect = np.abs(sub_data[key].sig['emgTA'].proc)

        # calculate enveloped signal using the calibrated proc signal
        sub_data[key].sig['emgSO'].envel = envelop(sub_data[key].sig['emgSO'].proc, fs)
        sub_data[key].sig['emgMG'].envel = envelop(sub_data[key].sig['emgMG'].proc, fs)
        sub_data[key].sig['emgLG'].envel = envelop(sub_data[key].sig['emgLG'].proc, fs)
        sub_data[key].sig['emgTA'].envel = envelop(sub_data[key].sig['emgTA'].proc, fs)

    return sub_data

def _calibrate_loadcell_signals(sub_info, sub_data):

    loadcell_offset_value = np.mean(sub_data['baseline'].sig['torque'].raw) * sub_info.scale_MVC_loadcell
    for key in sub_data.keys():
        sub_data[key].sig['torque'].calibrate(slope=sub_info.scale_MVC_loadcell, offset=loadcell_offset_value)

    return sub_data

def _remove_loadcell_offset_start_each_trial(sub_info, sub_data):

    for key in sub_data.keys():
        sub_data[key].sig['torque'].remove_offset(type_='start', val=int(sub_info.freq))

    return sub_data

def _find_MVC_normalize_torque_signals(sub_info, sub_data, plot_fig=plot_fig):

    torque = sub_data['mvc_vol'].sig['torque'].proc
    index_above_threshold = list(torque > 30) # set torque threshold at 30 Nm
    count = 0
    indexes = []
    # Extract indexes of torque data during the 5 MVC attempts (last 5 MVCs)
    for i in range(len(index_above_threshold)-1, 1, -1):
        if index_above_threshold[i]:
            indexes.append(i)
            if not index_above_threshold[i-1]:
                count += 1
                if count == 5:
                    break
    mvc_torque = max(torque[indexes])
    mvc_torque_index = np.argmax(torque[indexes])
    mvc_torque = (mvc_torque, mvc_torque_index) # only torques above threshold are indexed, not time series torques
    torques_above_threshold = torque[indexes]

    if plot_fig:
        fig = plt.figure(figsize=(11, 7))
        plt.subplot(1, 1, 1)
        plt.grid()
        plt.plot(torque[indexes], 'k')
        plt.plot(mvc_torque[1], mvc_torque[0] + 2, 'ro')
        plt.ylabel('Torque (Nm)')
        plt.tight_layout()
        plt.savefig('mvc_vol.png', dpi=300)
        shutil.move('mvc_vol.png', os.path.join('.', 'data', 'proc', sub_info.sub, 'mvc_vol.png'))
        plt.close()

    def _find_max_EMG(emg_signal, indexes):
        '''Define MVC EMG as the maximum EMG value in the 5 plantarflexion MVCs,
        not averaged over a window or tied to torque'''
        mvc_emg = max(emg_signal[indexes])
        mvc_emg_index = np.argmax(emg_signal[indexes])
        return mvc_emg, mvc_emg_index

    SO = sub_data['mvc_vol'].sig['emgSO'].envel
    MG = sub_data['mvc_vol'].sig['emgMG'].envel
    LG = sub_data['mvc_vol'].sig['emgLG'].envel
    mvc_SO = _find_max_EMG(SO, indexes)
    mvc_MG = _find_max_EMG(MG, indexes)
    mvc_LG = _find_max_EMG(LG, indexes)

    emgSO_above_threshold = sub_data['mvc_vol'].sig['emgSO'].proc[indexes]
    emgMG_above_threshold = sub_data['mvc_vol'].sig['emgMG'].proc[indexes]
    emgLG_above_threshold = sub_data['mvc_vol'].sig['emgLG'].proc[indexes]

    for i, val in enumerate(torque):
        if val > 40:            # Could break if torque goes above 40 during DF MVC
            last_index_for_TA = i
            break

    TA = sub_data['mvc_vol'].sig['emgTA'].envel
    TA_thresholded = list(TA[0:last_index_for_TA] > 0.05)
    indexes_for_TA_MVC = []
    current_run_True = []
    for i, val in enumerate(TA_thresholded):
        if val:
            current_run_True.append(i)
        else:
            if len(current_run_True) >= 500:
                indexes_for_TA_MVC.extend(current_run_True)
            current_run_True = []
    mvc_TA = _find_max_EMG(TA, indexes)

    Maximum_values = namedtuple('Maximum_values', 'mvc_torque mvc_SO mvc_MG mvc_LG mvc_TA')
    max_vals_and_indexes = Maximum_values(mvc_torque=mvc_torque,
                                        mvc_SO=mvc_SO,
                                        mvc_MG=mvc_MG,
                                        mvc_LG=mvc_LG,
                                        mvc_TA=mvc_TA)
    Above_threshold = namedtuple('Above_threshold', 'torques emgSO emgMG emgLG')
    signals_above_threshold = Above_threshold(torques=torques_above_threshold,
                                              emgSO=emgSO_above_threshold,
                                              emgMG=emgMG_above_threshold,
                                              emgLG=emgLG_above_threshold)

    for key in sub_data.keys():
        sub_data[key].sig['torque'].normalize(type_='value', value=max_vals_and_indexes.mvc_torque[0], signal_version='proc')
        # sub_data[key].sig['emgSO'].normalize(type_='value', value=max_vals_and_indexes.mvc_SO[0], signal_version='rect')
        # sub_data[key].sig['emgMG'].normalize(type_='value', value=max_vals_and_indexes.mvc_MG[0], signal_version='rect')
        # sub_data[key].sig['emgLG'].normalize(type_='value', value=max_vals_and_indexes.mvc_LG[0], signal_version='rect')
        # sub_data[key].sig['emgTA'].normalize(type_='value', value=max_vals_and_indexes.mvc_TA[0], signal_version='rect')

    return sub_data, max_vals_and_indexes, signals_above_threshold

def _determine_sit_rest_indexes(sub_info, sub_data, key):

    nsamples_before_trig = int(sub_info.freq * 0.5)
    idx1 = int(sub_data[key].sig['trig'].times[0] * sub_info.freq)
    idx2 = int(sub_data[key].sig['trig'].times[1] * sub_info.freq)
    if np.mean(sub_data[key].sig['torque'].proc[idx1 - nsamples_before_trig: idx1]) > \
            np.mean(sub_data[key].sig['torque'].proc[idx2 - nsamples_before_trig: idx2]):
        index_sit = idx1
        index_rest = idx2
    else:
        index_sit = idx2
        index_rest = idx1

    return index_rest, index_sit

def _calculate_activations(sub_info, sub_data, max_vals_and_indexes, plot_fig=plot_fig):

    def _calculate_peak_to_peak_amplitude(sub, key, freq, index, type, plot_fig=plot_fig):
        high_force_trials = ['90', '95', '100']
        if key in high_force_trials and type=='sit':
            nsamples = int(freq * 0.150) # Finds twitch peak at high force within 150 ms (could customise to 100 ms)
        else:
            nsamples = int(freq * 0.150) # Finds twitch peak at low-mod force within 150 ms
        index1, index2 = index, index + nsamples
        # find min and max torque, accounting for 15 ms electromechanical delay
        proc = sub_data[key].sig['torque'].proc # 50 Hz lowpass filtered
        time = sub_data[key].sig['torque'].times
        sig = proc[index1: index2]
        delay = int(0.010 * freq) # default: 0.015
        sig_after_delay = sig[delay:]
        time_after_delay = time[index1: index2][delay:]
        # find index and value of max torque
        index_max_nsamples, sig_max = np.where(sig == sig_after_delay.max())[0], sig_after_delay.max()
        # find index and value of min force in signal preceding the max force
        # sig_before_max = sig[:int(index_max_nsamples)] # signal is between stimulus and max torque
        # time_before_max = time[:int(index_max_nsamples)]
        sig_before_max = sig[delay: int(index_max_nsamples)] # signal is between EMD and max torque
        time_before_max = time[delay: int(index_max_nsamples)]
        if int(index_max_nsamples) == delay:
            index_min_nsamples, sig_min = index_max_nsamples, sig_max
        else:
            index_min_nsamples, sig_min = np.where(sig == sig_before_max.min())[0], sig_before_max.min()
        # calculate twitch amplitude
        signal_ptp = sig_max - sig_min
        if signal_ptp < 0:
            signal_ptp = 0
        index_min = index + int(index_min_nsamples)
        index_max = index + int(index_max_nsamples)
        # plot and check indexing of torque
        raw = sub_data[key].sig['torque'].raw  # torque in V
        loadcell_offset_value = np.mean(sub_data['baseline'].sig['torque'].raw) * sub_info.scale_MVC_loadcell
        raw = raw * sub_info.scale_MVC_loadcell - loadcell_offset_value  # torque in Nm
        raw = raw / max_vals_and_indexes.mvc_torque[0] * 100  # torque normalised to MVC
        raw = raw - np.mean(raw[:2000])  # unfiltered

        if plot_fig:
            plt.figure()
            plt.plot(time[index1: index2], raw[index1: index2], label='raw')
            # plt.plot(time[index1: index2], sig, label='filtered')
            plt.plot(time_after_delay, sig_after_delay, label='filt, after EMD')
            plt.plot(delay / freq + index1 / freq, sig[delay], 'ko', label='EMD={}ms'.format(int(delay / freq * 1000)))
            plt.plot(index_max_nsamples / freq + index1 / freq, sig_max, 'ro', label='max')
            plt.plot(time_before_max + index1 / freq, sig_before_max, label='filt, before max')
            plt.plot(index_min_nsamples / freq + index1 / freq, sig_min, 'go', label='min')
            plt.legend()
            plt.ylabel('Torque (%MVC)')
            plt.xlabel('Time within window (s)')
            if signal_ptp == 0:
                text = '{}, {}%; Max - min: {:.4} - {:.4} = {}'.format(sub_info.sub, key, sig_max, sig_min, signal_ptp)
            else:
                text = '{}, {}%; Max - min: {:.4} - {:.4} = {:.4}'.format(sub_info.sub, key, sig_max, sig_min, signal_ptp)
            plt.annotate(text, xy=(0.01, 1.01), xycoords='axes fraction', fontsize=8)
            plt.tight_layout()
            plt.savefig('sit_' + key + '.png', dpi=300)
            shutil.move('sit_' + key + '.png', os.path.join('.', 'data', 'proc', sub_info.sub, 'sit_' + key + '.png'))
            plt.close()
        return signal_ptp, index_min, index_max

    def _plot_signals(sub_data, sub_info, key, rest_idx1, rest_idx2, sit_idx1, sit_idx2, plot_fig=plot_fig):
        torque = sub_data[key].sig['torque'].proc
        emgSO = sub_data[key].sig['emgSO'].proc
        emgMG = sub_data[key].sig['emgMG'].proc
        emgLG = sub_data[key].sig['emgLG'].proc

        if plot_fig:
            fig = plt.figure(figsize=(11, 7))
            # torque
            plt.subplot(2, 1, 1)
            plt.grid()
            plt.plot(torque, 'k')
            rest_idxs, sit_idxs = np.arange(rest_idx1, rest_idx2), np.arange(sit_idx1, sit_idx2)
            rest_torque, sit_torque = torque[rest_idx1: rest_idx2], torque[sit_idx1: sit_idx2]
            plt.plot(rest_idxs, rest_torque, 'b', label='REST')
            plt.plot(sit_idxs, sit_torque, 'r', label='SIT')
            plt.legend()
            plt.autoscale(enable=True, axis='x', tight=True)
            text = sub_info.sub + ': ' + key + '%'
            plt.annotate(text, xy=(0, 1), xycoords='axes fraction', fontsize=8)
            plt.ylabel('Torque (%MVC)')
            # EMG
            plt.subplot(2, 1, 2)
            plt.grid()
            plt.plot(emgSO, 'k', label='SO')
            plt.plot(emgMG, 'g', label='MG')
            plt.plot(emgLG, 'b', label='LG')
            plt.legend()
            plt.ylabel('EMG (mV)')
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.tight_layout()
            plt.savefig(key + '.png', dpi=300)
            shutil.move(key + '.png', os.path.join('.', 'data', 'proc', sub_info.sub, key + '.png'))
            plt.close()

    print('\n' + sub_info.sub)
    activations = dict()
    for key in list(sub_data.keys())[4:]:
        torque = sub_data[key].sig['torque'].proc
        index_rest, index_sit = _determine_sit_rest_indexes(sub_info, sub_data, key) # , max_vals_and_indexes

        rest_ptp, rest_idx1, rest_idx2 = _calculate_peak_to_peak_amplitude(sub_info.sub, key, sub_info.freq, index_rest, type='rest')
        sit_ptp, sit_idx1, sit_idx2 = _calculate_peak_to_peak_amplitude(sub_info.sub, key, sub_info.freq, index_sit, type='sit')
        print(f'key: {key}%, SIT is after REST (1-75% MVC): {index_sit-index_rest > 0}')

        activation = (1 - (sit_ptp / rest_ptp)) * 100
        activation_ = {key: {'rest_ptp': rest_ptp, 'sit_ptp': sit_ptp, 'activation': activation}}
        activations.update(activation_)

        _plot_signals(sub_data, sub_info, key, rest_idx1, rest_idx2, sit_idx1, sit_idx2)

    return activations

def _calculate_torque_EMG_at_activations(sub_info, sub_data, plot_fig=plot_fig):

    nsamples_before_trig = int(sub_info.freq * 0.05) # mean EMG over 50 ms window
    torques_emgs = dict()

    for key in list(sub_data.keys())[4:]:
        index_rest, index_sit = _determine_sit_rest_indexes(sub_info, sub_data, key)
        # shift indexed EMG region away from filter artefact close to stimulus artefact
        filter_artefact_length = int(sub_info.freq * 0.05)
        index_start, index_stop = index_sit - (filter_artefact_length + nsamples_before_trig), index_sit - filter_artefact_length

        torque = np.mean(sub_data[key].sig['torque'].proc[index_start: index_stop])
        emgSO = np.mean(sub_data[key].sig['emgSO'].rect[index_start: index_stop])
        emgMG = np.mean(sub_data[key].sig['emgMG'].rect[index_start: index_stop])
        emgLG = np.mean(sub_data[key].sig['emgLG'].rect[index_start: index_stop])
        emgTA = np.mean(sub_data[key].sig['emgTA'].rect[index_start: index_stop])

        torques_emgs_ = {key: {'torque': torque, 'emgSO': emgSO, 'emgMG': emgMG, 'emgLG': emgLG, 'emgTA': emgTA}}
        torques_emgs.update(torques_emgs_)

        # plot and check indexing of EMG
        i = index_start - int(sub_info.freq * 0.01)
        j = index_sit + int(sub_info.freq * 0.01)

        if plot_fig:
            plt.figure()
            # EMG SO
            plt.subplot(3, 1, 1)
            emg = sub_data[key].sig['emgSO'].rect[i: j]
            time = sub_data[key].sig['emgSO'].times[i: j]
            plt.plot(time, emg, 'k')
            plt.plot(index_start / sub_info.freq, 0, 'g|', linewidth=5, label='start')
            plt.plot(index_stop / sub_info.freq, 0, 'r|', linewidth=5, label='stop')
            plt.ylabel('EMG SO (mV)')
            plt.legend()
            # EMG MG
            plt.subplot(3, 1, 2)
            emg = sub_data[key].sig['emgMG'].rect[i: j]
            time = sub_data[key].sig['emgMG'].times[i: j]
            plt.plot(time, emg, 'k')
            plt.plot(index_start / sub_info.freq, 0, 'g|', linewidth=5, label='start')
            plt.plot(index_stop / sub_info.freq, 0, 'r|', linewidth=5, label='stop')
            plt.ylabel('EMG MG (mV)')
            # EMG LG
            plt.subplot(3, 1, 3)
            emg = sub_data[key].sig['emgLG'].rect[i: j]
            time = sub_data[key].sig['emgLG'].times[i: j]
            plt.plot(time, emg, 'k')
            plt.plot(index_start / sub_info.freq, 0, 'g|', linewidth=5, label='start')
            plt.plot(index_stop / sub_info.freq, 0, 'r|', linewidth=5, label='stop')
            plt.ylabel('EMG LG (mV)')
            plt.xlabel('Time within window (s)')
            plt.tight_layout()
            plt.savefig('emg_' + key + '.png', dpi=300)
            shutil.move('emg_' + key + '.png', os.path.join('.', 'data', 'proc', sub_info.sub, 'emg_' + key + '.png'))
            plt.close()

    return torques_emgs

def _find_mmax_p1_idxs(sub_info, sub_data):
    # from the trial with increasing single pulse stimulation currents, index the last maximal stimulation
    idx = int(sub_data['max_curr'].sig['trig'].times[-1] * sub_info.freq)
    # index the first phase of the M wave within a 50 ms window, 2 ms after the stimulus
    ptp_start, ptp_stop = 0.002, 0.052 # in sec
    idx1 = int(idx + ptp_start * sub_info.freq)
    idx2 = int(idx1 + ptp_stop * sub_info.freq)
    return idx1, idx2

def _find_mmax_rms(sub_info, sub_data, idx1, idx2, muscle, plot_fig=plot_fig):
    # index the first phase of the M wave within a 50 ms window, 2 ms after the stimulus
    emg = sub_data['max_curr'].sig[muscle].proc
    time = sub_data['max_curr'].sig[muscle].times
    mmax = emg[idx1: idx2]

    # interpolate over the M wave
    xaxis = list(range(0, len(mmax)))
    f = interpolate.interp1d(xaxis, mmax)
    xaxis_new = np.arange(0, len(mmax) - 1, 0.1)
    mmax_new = f(xaxis_new)

    # identify the sample indexes where the first phase of the M wave crosses 0 volts
    # similarly to Thomas C (1997) Fatigue in human thenar muscles paralysed by spinal cord injury
    min_val = abs(min(mmax_new))
    max_val = max(mmax_new)
    height = np.mean([min_val, max_val]) * .7
    indexes, _ = scipy.signal.find_peaks(abs(mmax_new), height=height, distance=5)
    if plot_fig:
        plt.plot(mmax_new,'.-')
        plt.plot(indexes, mmax_new[indexes], 'ro', label='min, max')

    # reference the index of the first of the two peaks
    peak_index = indexes[0]
    if mmax_new[peak_index] < 0:
        mmax_new *= -1
    for i in range(peak_index, 0, -1):
        # find the zero-crossing to the left of the first peak
        if mmax_new[i] > 0 and mmax_new[i - 1] < 0:
            idx_start_p1_mmax = i - 1
            break
        else:
            idx_start_p1_mmax = 0
    for i in range(peak_index, len(mmax_new)):
        # find the zero-crossing to the right of the first peak
        if mmax_new[i] > 0 and mmax_new[i + 1] < 0:
            idx_stop_p1_mmax = i + 1
            break

    # print(idx_start_p1_mmax, idx_stop_p1_mmax)
    if plot_fig:
        plt.plot([idx_start_p1_mmax, idx_stop_p1_mmax],
                 [mmax_new[idx_start_p1_mmax], mmax_new[idx_stop_p1_mmax]],
                 'bo', label='cross 0V')
        plt.legend()
        plt.xlabel('Samples')
        plt.ylabel('EMG (mV)')
        plt.tight_layout()
        plt.savefig('mmax-p1-' + muscle + '.png', dpi=300)
        shutil.move('mmax-p1-' + muscle + '.png', os.path.join('.', 'data', 'proc', sub_info.sub, 'mmax-p1-' + muscle + '.png'))
        plt.close()

    # calculate the root-mean-square of the first phase of the M wave
    # from sklearn.metrics import mean_squared_error  # test RMS function
    # i = np.zeros(len(mmax_new[idx_start_p1_mmax: idx_stop_p1_mmax]))
    # np.sqrt(mean_squared_error(i, mmax_new[idx_start_p1_mmax: idx_stop_p1_mmax])) # gets same answer
    # mmax_p1_rms = np.sqrt(np.sum(mmax_new[idx_start_p1_mmax: idx_stop_p1_mmax] ** 2) / len(mmax_new[idx_start_p1_mmax: idx_stop_p1_mmax]))
    mmax_p1_rms = np.sqrt(np.mean(mmax_new[idx_start_p1_mmax: idx_stop_p1_mmax] ** 2))

    # check the root-mean-square against mean of the rectified signal
    mmax_p1_avrect = np.mean(abs(mmax_new[idx_start_p1_mmax: idx_stop_p1_mmax]))

    return mmax_p1_rms, mmax_p1_avrect, mmax_new, idx_start_p1_mmax, idx_stop_p1_mmax

def _find_mvc_emg_rms(sub_info, max_vals_and_indexes, signals_above_threshold, muscle, plot_fig=plot_fig, rms_type='MVC emg at MVC torque'):
    # read in signals
    torque = signals_above_threshold.torques
    emgSO = signals_above_threshold.emgSO # processed EMG (not rectified or enveloped)
    emgMG = signals_above_threshold.emgMG
    emgLG = signals_above_threshold.emgLG

    if muscle == 'emgSO':
        emg = emgSO
    elif muscle == 'emgMG':
        emg = emgMG
    elif muscle == 'emgLG':
        emg = emgLG

    # get max value and index of rectified EMG, and calculate RMS from a 50 ms window over it
    if rms_type == 'RMS over max rect EMG':
        # print(rms_type)
        half_win = int(sub_info.freq * 0.05 / 2)
        # rectify signal, find the maximal instance of EMG
        emg_rect = abs(emg)
        mvc_emg_rect = np.max(emg_rect)
        # use the index at maximal EMG, calculate RMS of 50 ms window over the index
        mvc_emg_idx = np.where(emg_rect == mvc_emg_rect)[0][0]
        mvc_emg_rms = np.sqrt(np.mean(emg[mvc_emg_idx - half_win: mvc_emg_idx + half_win] ** 2))

        if plot_fig:
            plt.subplot(1,1,1)
            samples = list(range(0, len(emg)))
            plt.plot(samples, emg_rect, 'k')
            plt.plot(samples[mvc_emg_idx - half_win: mvc_emg_idx + half_win], emg_rect[mvc_emg_idx - half_win: mvc_emg_idx + half_win], 'r')
            plt.xlabel('Samples')
            plt.ylabel('EMG (mV)')
            plt.tight_layout()
            filename = 'mvc_rms_emg_' + muscle.split('emg')[1] + '.png'
            plt.savefig(filename, dpi=300)
            shutil.move(filename, os.path.join('.', 'data', 'proc', sub_info.sub, filename))
            plt.close()

        return mvc_emg_rms

    # use moving window, calculate MVC EMG from peak RMS of the EMG signals during the 5 MVCs
    if rms_type == 'moving window RMS':
        half_win = int(sub_info.freq * 0.05 / 2)
        emg_rms = np.zeros(emg.size)
        # Loop through and compute normalised moving window.
        # Window is smaller at the start and the end of the signal.
        for i in range(emg_rms.size - 1):
            if i < half_win:
                emg_rms[i] = np.sqrt(np.mean(emg[0: i + half_win] ** 2))
            elif i > emg_rms.size - half_win:
                emg_rms[i] = np.sqrt(np.mean(emg[i - half_win: emg_rms.size - 1] ** 2))
            else:
                emg_rms[i] = np.sqrt(np.mean(emg[i - half_win: i + half_win] ** 2))

        mvc_emg_rms = np.max(emg_rms)
        mvc_emg_idx = np.where(emg_rms == mvc_emg_rms)[0][0]

        if plot_fig:
            plt.subplot(1,1,1)
            plt.plot(emg, 'k')
            plt.plot(emg_rms, 'y')
            plt.plot(mvc_emg_idx, mvc_emg_rms, 'ro')
            plt.xlabel('Samples')
            plt.ylabel('EMG (mV)')
            plt.tight_layout()
            filename = 'mvc_rms_emg_' + muscle.split('emg')[1] + '.png'
            plt.savefig(filename, dpi=300)
            shutil.move(filename, os.path.join('.', 'data', 'proc', sub_info.sub, filename))
            plt.close()

        return mvc_emg_rms

    # calculate MVC EMG at maximal torque
    if rms_type == 'MVC emg at MVC torque':
        # read index and values of MVC torque
        # shift index back by 50 ms, to be comparable with trial EMG that is 50 ms before supramax stim
        filter_artefact_length = int(sub_info.freq * 0.05)
        mvc_torque_idx = max_vals_and_indexes.mvc_torque[1] - filter_artefact_length
        mvc_torque = max_vals_and_indexes.mvc_torque[0]
        # get root mean square EMG over 50 ms window over the MVC index
        half_win = int(sub_info.freq * 0.05 / 2)
        mvc_indexes = list(range(mvc_torque_idx - half_win, mvc_torque_idx + half_win))
        mvc_emg = emg[mvc_torque_idx - half_win: mvc_torque_idx + half_win]
        mvc_emg_rms = np.sqrt(np.mean(mvc_emg ** 2))

        if plot_fig:
            plt.subplot(2,1,1)
            plt.plot(torque, 'k')
            plt.plot(mvc_torque_idx, mvc_torque, 'ro')
            plt.ylabel('Torque (Nm)')

            plt.subplot(2,1,2)
            plt.plot(emg, 'k')
            # plt.plot(mvc_torque_idx, emg[mvc_torque_idx], 'ro')
            plt.plot(mvc_indexes, mvc_emg, 'r')
            plt.xlabel('Samples')
            plt.ylabel('EMG (mV)')
            plt.tight_layout()
            filename = 'mvc_rms_emg_' + muscle.split('emg')[1] + '.png'
            plt.savefig(filename, dpi=300)
            shutil.move(filename, os.path.join('.', 'data', 'proc', sub_info.sub, filename))
            plt.close()

        return mvc_emg_rms

def _find_time_difference_btw_mvcEMG_mvcTorque(sub_info, max_vals_and_indexes, signals_above_threshold, muscle, plot_fig=plot_fig):
    # read in signals
    torque = signals_above_threshold.torques
    emgSO = signals_above_threshold.emgSO  # processed EMG (not rectified or enveloped)
    emgMG = signals_above_threshold.emgMG
    emgLG = signals_above_threshold.emgLG

    if muscle == 'emgSO':
        emg = emgSO
    elif muscle == 'emgMG':
        emg = emgMG
    elif muscle == 'emgLG':
        emg = emgLG

    # find index, value of MVC torque
    mvc_torque_idx = max_vals_and_indexes.mvc_torque[1]
    mvc_torque = max_vals_and_indexes.mvc_torque[0]

    # find index, value of max rect EMG
    emg_rect = abs(emg)
    mvc_emg_rect = np.max(emg_rect)
    mvc_emg_rect_idx = np.where(emg_rect == mvc_emg_rect)[0][0]

    # find index, value of max RMS from moving window
    half_win = int(sub_info.freq * 0.05 / 2)
    emg_rms = np.zeros(emg.size)
    for i in range(emg_rms.size - 1):
        if i < half_win:
            emg_rms[i] = np.sqrt(np.mean(emg[0: i + half_win] ** 2))
        elif i > emg_rms.size - half_win:
            emg_rms[i] = np.sqrt(np.mean(emg[i - half_win: emg_rms.size - 1] ** 2))
        else:
            emg_rms[i] = np.sqrt(np.mean(emg[i - half_win: i + half_win] ** 2))

    mvc_emg_mvwin_rms = np.max(emg_rms)
    mvc_emg_mvwin_idx = np.where(emg_rms == mvc_emg_mvwin_rms)[0][0]

    # calculate differences in time
    freq = sub_info.freq
    diff_maxRect_torque = (mvc_emg_rect_idx - mvc_torque_idx) / freq
    diff_mvwin_torque = (mvc_emg_mvwin_idx - mvc_torque_idx) / freq

    if plot_fig:
        time = np.arange(0, len(torque) / freq, 1 / freq)
        plt.subplot(3, 1, 1)
        plt.plot(torque, 'k')
        plt.plot(mvc_torque_idx, mvc_torque, 'ro')
        plt.ylabel('Torque (Nm)')

        plt.subplot(3, 1, 2)
        plt.plot(emg_rect, 'k')
        plt.plot(mvc_emg_rect_idx, mvc_emg_rect, 'ro')
        plt.ylabel('Rect EMG (mV)')

        plt.subplot(3, 1, 3)
        plt.plot(emg, 'k')
        plt.plot(emg_rms, 'y')
        plt.plot(mvc_emg_mvwin_idx, mvc_emg_mvwin_rms, 'ro')
        plt.xlabel('Samples')
        plt.ylabel('RMS EMG (mV)')
        plt.tight_layout()
        filename = 'mvc_emg_torque_times_' + muscle.split('emg')[1] + '.png'
        plt.savefig(filename, dpi=300)
        shutil.move(filename, os.path.join('.', 'data', 'proc', sub_info.sub, filename))
        plt.close()

    return diff_maxRect_torque, diff_mvwin_torque

def _find_trial_emg_rms(sub_info, sub_data, plot_fig=plot_fig):
    nsamples_before_trig = int(sub_info.freq * 0.05)  # get EMG over 50 ms window
    emgs_rect = dict()
    emgs_rms = dict()

    i = 1
    j = len(list(sub_data.keys())[4:])

    def _determine_sit_rest_indexes(sub_info, sub_data, key):

        nsamples_before_trig = int(sub_info.freq * 0.5)
        idx1 = int(sub_data[key].sig['trig'].times[0] * sub_info.freq)
        idx2 = int(sub_data[key].sig['trig'].times[1] * sub_info.freq)
        if np.mean(sub_data[key].sig['torque'].proc[idx1 - nsamples_before_trig: idx1]) > \
                np.mean(sub_data[key].sig['torque'].proc[idx2 - nsamples_before_trig: idx2]):
            index_sit = idx1
            index_rest = idx2
        else:
            index_sit = idx2
            index_rest = idx1

        return index_rest, index_sit

    for key in list(sub_data.keys())[4:]:
        index_rest, index_sit = _determine_sit_rest_indexes(sub_info, sub_data, key)
        # shift indexed EMG region away from filter artefact close to stimulus artefact
        filter_artefact_length = int(sub_info.freq * 0.05)
        index_start, index_stop = index_sit - (filter_artefact_length + nsamples_before_trig), index_sit - filter_artefact_length

        emgSO_rect = sub_data[key].sig['emgSO'].rect[index_start: index_stop]
        emgMG_rect = sub_data[key].sig['emgMG'].rect[index_start: index_stop]
        emgLG_rect = sub_data[key].sig['emgLG'].rect[index_start: index_stop]
        emgs_rect_ = {key: {'emgSO': emgSO_rect, 'emgMG': emgMG_rect, 'emgLG': emgLG_rect}}
        emgs_rect.update(emgs_rect_)

        emgSO_proc = sub_data[key].sig['emgSO'].proc[index_start: index_stop]
        emgMG_proc = sub_data[key].sig['emgMG'].proc[index_start: index_stop]
        emgLG_proc = sub_data[key].sig['emgLG'].proc[index_start: index_stop]
        emgSO_rms = np.sqrt(np.mean(emgSO_proc ** 2))
        emgMG_rms = np.sqrt(np.mean(emgMG_proc ** 2))
        emgLG_rms = np.sqrt(np.mean(emgLG_proc ** 2))
        emgs_rms_ = {key: {'emgSO': emgSO_rms, 'emgMG': emgMG_rms, 'emgLG': emgLG_rms}}
        emgs_rms.update(emgs_rms_)

        if plot_fig:
            plt.subplot(j, 1, i)
            plt.plot(emgSO_proc, 'k', label='SO')
            plt.plot(emgMG_proc, 'r', label='MG')
            plt.plot(emgLG_proc, 'b', label='LG')
            plt.ylim(-0.4, 0.4)
            plt.yticks(ticks=[], labels=[])
            if i == 2:
                plt.legend()
            if i == 6:
                plt.ylabel('EMG (ylim 0-0.2 mV); SO (k), MG (r), LG (b)')
            i += 1

    if plot_fig:
        plt.xlabel('Samples')
        plt.tight_layout()
        plt.savefig('trial_rms_emg.png', dpi=300)
        shutil.move('trial_rms_emg.png', os.path.join('.', 'data', 'proc', sub_info.sub, 'trial_rms_emg.png'))
        plt.close()

    return emgs_rect, emgs_rms

def _normalise_emg(sub_data, mvc_emg_rms, mmax_p1_rms, emgs_rms, muscle):
    emg_norm_mvc, emg_norm_mmax = dict(), dict()
    for key in list(sub_data.keys())[4:]:
        emg_rms = emgs_rms[key][muscle]

        emg_mvc = emg_rms / mvc_emg_rms * 100
        emg_norm_mvc_ = {key: {'norm_mvc': emg_mvc}}
        emg_norm_mvc.update(emg_norm_mvc_)

        emg_mmax = emg_rms / mmax_p1_rms * 100
        emg_norm_mmax_ = {key: {'norm_mmax': emg_mmax}}
        emg_norm_mmax.update(emg_norm_mmax_)

    return emg_norm_mvc, emg_norm_mmax

def _calc_slope_of_line(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - (slope * x1)
    # y = slope * x + intercept
    return slope, intercept

def _calc_lower_limit(samples, slope, intercept):
    y = (slope * samples) + intercept
    return y