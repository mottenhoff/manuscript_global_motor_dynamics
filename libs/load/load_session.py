import bisect
import logging
from dataclasses import dataclass
from datetime import datetime
from os.path import getctime 
from pathlib import Path

import numpy as np

from libs.load.read_xdf import xdf_to_dict
from libs.load import load_locations

@dataclass
class Session:
    ppt_id: int
    type: str
    eeg: np.array
    ts: np.array
    fs: int
    anatomical_location: dict
    channels: np.array
    trial_nums: np.array
    trial_names: np.array

def _get_created_date(file, dt_format='%Y%m%d%H%M%S'):
    # Returns the formatted date of creation of a file
    return datetime.fromtimestamp(getctime(file)).strftime(dt_format)

def _locate_pos(available_tss, target_ts):
    # Locate the the closest index within a list of indices

    pos = bisect.bisect_right(available_tss, target_ts)
    if pos == 0:
        return 0

    if pos == len(available_tss):
        return len(available_tss)-1

    if abs(available_tss[pos]-target_ts) < abs(available_tss[pos-1]-target_ts):
        return pos
    else:
        return pos-1

def _get_trials_info(eeg, eeg_ts, markers, marker_ts):
    # Create a label and trial numbers per timestamp

    # Find which markers correspond to the start and end of a trial
    trial_start_mask = [marker[0].split(';')[0]=='start' for marker in markers]
    trial_end_mask =   [marker[0].split(';')[0]=='end' for marker in markers]

    # Find the indices corresponding to the start and end of the trial
    trial_idc_start = np.array([_locate_pos(eeg_ts, trial) for trial in marker_ts[trial_start_mask]])
    trial_idc_end =   np.array([_locate_pos(eeg_ts, trial) for trial in marker_ts[trial_end_mask]])

    # Retrieve the corresponding labels per trial
    trial_labels = [marker[0].split(';')[1] for marker in markers if marker[0].split(';')[0] == 'start']

    # Map the label and trial number per index.
    trial_seq =  [0] * eeg.shape[0] # Trial labels sequential
    trial_nums = [0] * eeg.shape[0]
    for i, idx_start in enumerate(trial_idc_start):
        trial_seq[ idx_start:trial_idc_end[i]] = [trial_labels[i]] * (trial_idc_end[i]-idx_start)
        trial_nums[idx_start:trial_idc_end[i]] = [i] * (trial_idc_end[i]-idx_start)

    return np.array(trial_seq), np.array(trial_nums)

def _get_experiment_data(result):
    # TODO: Offset markers? (see load_grasp_data)
    marker_idx_exp_start = result['GraspMarkerStream']['data'].index(['experimentStarted'])
    marker_idx_exp_end =   result['GraspMarkerStream']['data'].index(['experimentEnded'])

    eeg_idx_exp_start = _locate_pos(result['Micromed']['ts'], 
                                    result['GraspMarkerStream']['ts'][marker_idx_exp_start])
    eeg_idx_exp_end =   _locate_pos(result['Micromed']['ts'],
                                    result['GraspMarkerStream']['ts'][marker_idx_exp_end])

    eeg =    result['Micromed']['data'][eeg_idx_exp_start:eeg_idx_exp_end, :]
    eeg_ts = result['Micromed']['ts'][eeg_idx_exp_start:eeg_idx_exp_end]

    marker =    result['GraspMarkerStream']['data'][marker_idx_exp_start:marker_idx_exp_end]
    marker_ts = result['GraspMarkerStream']['ts'][marker_idx_exp_start:marker_idx_exp_end]

    return eeg, eeg_ts, marker, marker_ts

def load_grasp_seeg(file):
    ''' Loads xdf file and returns a dict with all necessary information'''
    
    file = Path(file)
    logging.info(f'Loading file: {file}')

    result, _ = xdf_to_dict(file)

    eeg, eeg_ts, markers, markers_ts = _get_experiment_data(result)
    trials, trial_nums = _get_trials_info(eeg, eeg_ts, markers, markers_ts)

    label_map = {'0': 'rest', 'Links': 'left', 'Rechts': 'right'}
    trials = np.vectorize(label_map.get)(trials)

    seeg = {}
    seeg['subject'] =           file.parent
    seeg['experiment_type'] =   file.stem
    seeg['channel_names'] =     result['Micromed']['channel_names']
    seeg['eeg'] =               eeg.astype(np.float64)
    seeg['eeg_ts'] =            eeg_ts
    seeg['trial_labels'] =      trials
    seeg['trial_numbers'] =     trial_nums
    seeg['fs'] =                result['Micromed']['fs']
    seeg['dtype'] =             result['Micromed']['data_type']
    seeg['first_ts'] =          result['Micromed']['first_ts']
    seeg['last_ts'] =           result['Micromed']['last_ts']
    seeg['total_stream_time'] = result['Micromed']['total_stream_time']
    seeg['samplecount'] =       result['Micromed']['sample_count']

    return seeg

def load(paths):
    path_eeg, path_locs = paths

    seeg = load_grasp_seeg(path_eeg)
    locs = load_locations.load(path_locs)

    s = Session(
        ppt_id =      seeg['subject'],
        type =        seeg['experiment_type'],
        eeg =         seeg['eeg'],
        ts =          seeg['eeg_ts'],
        fs =          seeg['fs'],
        channels =    seeg['channel_names'],
        trial_nums =  seeg['trial_numbers'],
        trial_names = seeg['trial_labels'],
        anatomical_location = locs,
    )

    return s