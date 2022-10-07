import os
from collections import namedtuple
import spike2py as spk2


def read_subject_log(path, sub):
    """Read subject's log file and return named tuple"""
    f = os.path.join(path, 'log.txt')
    with open(f, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split(':')
            if line[0] == 'age':
                age = float(line[-1])
            elif line[0] == 'sex':
                sex = line[-1]
            elif line[0] == 'height (m)':
                height = float(line[-1])
            elif line[0] == 'weight (kg)':
                weight = float(line[-1])
            elif line[0] == 'sampling rate':
                freq = float(line[-1])
            elif line[0] == 'scale MVC load cell':
                scale_MVC_loadcell = float(line[-1])
            elif line[0] == 'Activations':
                activations_baseline = [float(i) for i in line[-1].strip().split(' ')]
    Subject_Info = namedtuple('Subject_info', 'sub age sex height weight freq scale_MVC_loadcell activations_baseline')
    return Subject_Info(sub=sub, age=age, sex=sex, height=height, weight=weight,
                        freq=freq, scale_MVC_loadcell=scale_MVC_loadcell, activations_baseline=activations_baseline)

def generate_spike2py_signalInfos(sub):
    """Generate spike2py signal info for all signals, returned as a list"""
    torque_info = spk2.SigInfo(name='torque', stype='waveform', s2name='forceMVC',
                               filt_cutoff=100, filt_order=4, filt_type='lowpass')
    # On import, EMG are bandpass filtered 20-450 Hz with spike2py default
    emgSO_info = spk2.SigInfo(name='emgSO', stype='sEMG', s2name='emgSOL',
                              offset_type='start', offset_val=int(sub.freq), offset_sig='raw')
    emgMG_info = spk2.SigInfo(name='emgMG', stype='sEMG', s2name='emgMG',
                              offset_type='start', offset_val=int(sub.freq), offset_sig='raw')
    emgLG_info = spk2.SigInfo(name='emgLG', stype='sEMG', s2name='emgLG',
                              offset_type='start', offset_val=int(sub.freq), offset_sig='raw')
    emgTA_info = spk2.SigInfo(name='emgTA', stype='sEMG', s2name='emgTA',
                              offset_type='start', offset_val=int(sub.freq), offset_sig='raw')
    trig_info = spk2.SigInfo(name='trig', stype='trig', s2name='stim')
    return [torque_info, emgSO_info, emgMG_info, emgLG_info, emgTA_info, trig_info]

def generate_subject_list():
    return ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub08', 'sub09', 'sub10',
            'sub11', 'sub12', 'sub13', 'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19', 'sub20',
            'sub21', 'sub22', 'sub23', 'sub24', 'sub25']
    # return ['sub01', 'sub02']
