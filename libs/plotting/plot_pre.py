from itertools import product

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, f_oneway, ttest_rel
# import tensortools as tt

import libs.windowing as windowing

CHANNELS = 2

def plot_pca_over_all_data(session, savepath):

    eeg = (session.eeg - session.eeg.mean(axis=0)) / session.eeg.std(axis=0)

    pca = PCA()
    pca.fit(eeg)
    components = pca.transform(eeg)

    # components = pca.components_
    ev = pca.explained_variance_
    evr = pca.explained_variance_ratio_

    fig, ax = plt.subplots(3, 3)

    n = 5
    ax[0, 0].set_title(f'First {n} components')
    ax[0, 0].plot(components[:, :5])
    ax[0, 0].set_xlabel('Component')
    ax[0, 0].set_ylabel('a.u.')    

    ax[0, 1].set_title('Explained variance')    
    ax[0, 1].plot(ev)
    ax[0, 1].set_xlabel('Component')
    ax[0, 1].set_ylabel('Variance')    

    ax[0, 2].set_title('Explained variance ratio')
    ax[0, 2].plot(evr)
    ax[0, 2].set_xlabel('Component')
    ax[0, 2].set_ylabel('Variance [% / total]')    
    
    ax[1, 0].set_title('PC 0 vs PC 1')
    ax[1, 0].plot(components[0], components[1])
    
    ax[1, 1].set_title('PC 1 vs PC 2')
    ax[1, 1].plot(components[1], components[2])

    ax[1, 2].set_title('PC 2 vs PC 3')
    ax[1, 2].plot(components[2], components[3])

    ax[2, 0].set_title('PC 3 vs PC 4')
    ax[2, 0].plot(components[3], components[4])
    
    ax[2, 1].set_title('PC 4 vs PC 5')
    ax[2, 1].plot(components[4], components[5])

    ax[2, 2].set_title('PC 5 vs PC 6')
    ax[2, 2].plot(components[5], components[6])

    fig.tight_layout()    
    fig.savefig('pca.png')

    return fig, ax

def tensor_component_analysis(session, savepath):
    # Requires chosing the amount of components (per ppt? Or equal?)
    # What is on the Y-axes
    # Color trials by label
    # Make bars in neurons and sort by electrode (if not already). Can also sort by physical distance

    # Normalize


    # Input = Neuron x Time x Trials
    trials = session.trials.transpose(2, 1, 0)
    original_shape = trials.shape
    trials = normalize(trials.reshape(trials.shape[0], -1), axis=0)
    trials = trials.reshape(original_shape)

    ensemble = tt.Ensemble(fit_method='ncp_hals')
    ensemble.fit(trials, ranks=range(1, 9), replicates=4)

    fig, axes = plt.subplots(1, 2)
    tt.plot_objective(ensemble, ax=axes[0])   # plot reconstruction error as a function of num components.
    tt.plot_similarity(ensemble, ax=axes[1])  # plot model similarity as a function of num components.
    fig.tight_layout()
    # Plot the low-d factors for an example model, e.g. rank-2, first optimization run / replicate.
    num_components = 3
    replicate = 0
    tt.plot_factors(ensemble.factors(num_components)[replicate])  # plot the low-d factors

    plt.savefig(f'{session.ppt_id}_test.png')

    return

def principal_component_analysis(session, savepath):


    plot_pca_over_all_data(session, savepath)

    eeg = (session.eeg - session.eeg.mean(axis=0)) / session.eeg.std(axis=0)

    pca = PCA().fit(eeg)
    pcs = pca.transform(eeg)



    return 

def plot_average_trials(session, savepath):

    # Window all trials
    ts_per_trial = np.arange(0, session.trials.shape[1] / session.fs, 1/session.fs)
    trials = [windowing.window(trial, ts_per_trial, 100, 100, session.fs).mean(axis=1) 
              for trial in session.trials]
    trials = np.stack(trials)
        
    # Split per class
    is_move_trial = np.where(session.trial_labels=='rest', False, True)

    rest, move = trials[~is_move_trial, :, :], trials[is_move_trial, :, :]

    # Get significantly modulated channels
    n_windows, n_channels = trials.shape[1:]
    significance = np.empty((n_windows, n_channels))

    for window_num, channel_num in product(np.arange(n_windows), np.arange(n_channels)):
        
        t, p = ttest_rel(rest[:, window_num, channel_num], 
                         move[:, window_num, channel_num])
        p *= significance.size  # Bonferroni

        significance[window_num, channel_num] = p

    # Print anatomical location of significant channels.
    significant_channel_names = session.channels[np.unique(np.where(significance < 0.05)[1])]

    for ch in significant_channel_names:
        print(session.anatomical_location[ch])

    # Plot the traces of these significantly modulated channels
    electrodes, counts = np.unique([ch.strip('0123456789') for ch in significant_channel_names],
                                   return_counts=True)

    # Plotting
    fig, axs = plt.subplots(nrows=counts.max(),
                            ncols=electrodes.size,
                            figsize=(16, 12))

    for ax in axs.flatten():
        ax.axis('off')

    n_ticks, n_seconds_per_trial = 4, 3
    xticks = np.linspace(0, trials.shape[1], n_ticks)
    xtickslabels = np.linspace(0, n_seconds_per_trial, n_ticks)

    for electrode_num, electrode in enumerate(electrodes):
        
        channels_in_electrode = [ch_name for ch_name in significant_channel_names if electrode in ch_name]

        for channel_num, channel in enumerate(channels_in_electrode):
            axs[channel_num, electrode_num].axis('on')

            ch_idx = np.where(session.channels == channel)[0]
            print(electrode_num, channel_num, ch_idx)

            rest_mean, move_mean = rest[:, :, ch_idx].mean(axis=0), move[:, :, ch_idx].mean(axis=0)
            rest_mean -= rest_mean[0, :]
            move_mean -= move_mean[0, :]


            axs[channel_num, electrode_num].plot(rest_mean, color='black')
            axs[channel_num, electrode_num].plot(move_mean, color='orange')

            # Color sig windows
            is_sig_window = np.where(significance[:, ch_idx] < 0.05)[0]
            bottom, top = axs[channel_num, electrode_num].get_ylim()
            # is_sig_window = np.where(significance[:, ch_idx] < 0.05, True, False)

            for sig_window in is_sig_window:

                axs[channel_num, electrode_num].fill_between(
                    x=[sig_window-0.5, sig_window+0.5],
                    y1=bottom,
                    y2=top,
                    color='red')


            # axs[channel_num, electrode_num].fill_between(
            #     x=np.arange(rest_mean.shape[0]),
            #     y1=bottom,
            #     y2=top,
            #     where=is_sig_window.ravel(),
            #     color='red')

            # Makeup
            anatomical_location = session.anatomical_location.get(channel, None)
            axs[channel_num, electrode_num].set_title(anatomical_location if channel_num != 0 else f'Electrode {electrode}\n{anatomical_location}')
            axs[channel_num, electrode_num].spines[['right', 'top', 'bottom', 'left']].set_visible(False)

            axs[channel_num, electrode_num].set_xticks(xticks)
            axs[channel_num, electrode_num].set_xticklabels(xtickslabels)

            if electrode_num == 0 and channel_num==0:
                axs[channel_num, electrode_num].set_ylabel('Power')
                axs[channel_num, electrode_num].set_xlabel('Time[s]')


    fig.tight_layout()
    fig.savefig('tmp.png')

    return
            # axs[channel_num, electrode_num].fill_between(
            #                                         np.arange(m.shape[1]),
            #                                         (m.mean(axis=0) - m.std(axis=0)).flatten(),
            #                                         (m.mean(axis=0) + m.std(axis=0)).flatten(),
            #                                         alpha=.3,
            #                                         color='orange')

def make_all(session, savepath):

    plot_average_trials(session, savepath)    
    # tensor_component_analysis(session, savepath)
    principal_component_analysis(session, savepath)
    
    pass