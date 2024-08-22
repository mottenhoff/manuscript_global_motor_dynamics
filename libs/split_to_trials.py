import numpy as np

def c2t(session):

    trials_idc = []
    labels = []
    start_label = None
    start_idx = None
    prev = None
    for curr_idx, curr in enumerate(session.trial_names):
        
        if start_label == None:
            start_label = curr

        if curr != start_label and prev == start_label:
            # Detected change of labels
            current_idc = [start_idx, curr_idx]

            if None not in current_idc:
                trials_idc += [current_idc]
                labels += [start_label]
            
            start_idx = curr_idx
            start_label = curr
        
        prev = curr

    window_sizes = [np.diff(idc) for idc in trials_idc]
    min_samples = min(window_sizes)[0]
    max_samples = max(window_sizes)[0]
    print(f'Reduced trial size to {min_samples} samples. (max seen window size: {max_samples})')

    trials = []
    for idc in trials_idc:
        trial = session.eeg[idc[0]:idc[1], :]
        trials += [trial[0:min_samples, :]]

    session.trials =       np.array(trials)
    session.trial_labels = np.array(labels)

    return session