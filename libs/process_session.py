import numpy as np
import scipy.signal
from scipy.fftpack import next_fast_len
from mne.filter import filter_data

from libs.load.load_yaml import load_yaml
from libs.data_quality.check_quality import QualityChecker

c = load_yaml('./config.yml')

def hilbert_transform(x):
    # x.shape == [samples x channels]

    n_fourier_components = next_fast_len(x.shape[0])

    y = scipy.signal.hilbert(x, n_fourier_components, axis=0)[:x.shape[0]]

    return np.abs(y)

def frequency_filters(session, filters):

    eeg = session.eeg - session.eeg.mean(axis=0)
    eeg = scipy.signal.detrend(eeg, axis=0)

    eeg = [filter_data(eeg.T, session.fs, 
                       l_freq = cutoffs[0] if cutoffs[0] > 0 else None,
                       h_freq = cutoffs[1])
           for band, cutoffs in filters.items()]

    session.eeg = np.vstack(eeg).T
    
    return session

def signal_quality(session):

    qc = QualityChecker()

    if any(qc.consistent_timestamps(session.ts, session.fs)):
        print('Invalid Timesteps!')
        return None

    irrelevant_channels = np.hstack([
        qc.get_disconnected_channels(session.eeg, session.channels),
        qc.get_ekg_channel(session.eeg, session.channels),
        qc.get_marker_channels(session.eeg, session.channels)
    ]).astype(int)

    irrelevant_channels = np.append(irrelevant_channels, 
                                    [i for i, chs in enumerate(session.channels) if '+' in chs])

    session.eeg =      np.delete(session.eeg, irrelevant_channels, axis=1)
    session.channels = np.delete(session.channels, irrelevant_channels)

    return session

def process_session(session, filters):

    session = signal_quality(session)

    if not session:
        return 'invalid'

    session = frequency_filters(session, filters)
    session.eeg = hilbert_transform(session.eeg)

    return session

