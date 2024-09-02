from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from libs.load.load_yaml import load_yaml

c = load_yaml('./config.yml')

def raw_eeg(session, savepath):
    # including markers

    n_channels = session.eeg.shape[1]
    n_rows = 20

    idc_per_col = np.array_split(np.arange(n_channels), n_channels/n_rows)
    max_rows = max([len(idc) for idc in idc_per_col])

    fig, ax = plt.subplots(nrows=max_rows, ncols=len(idc_per_col), 
                           dpi=200, figsize=(24, 16))
    
    for col_i, ax_col in enumerate(idc_per_col):
        for row_i, ch_num in enumerate(ax_col):
            ax[row_i, col_i].plot(session.eeg[:, ch_num], linewidth=1, color='k')
            ax[row_i, col_i].spines[['top', 'right', 'left', 'bottom']].set_visible(False)
            ax[row_i, col_i].tick_params(axis='x', which='both', 
                                         bottom=False, labelbottom=False)
            ax[row_i, col_i].tick_params(axis='y', which='both',
                                         left=False, labelleft=False)
            ax[row_i, col_i].set_ylabel(session.channels[ch_num])
            ax[row_i, col_i].set_ylim(-1000, 1000)


    fig.suptitle('Raw signal')
    fig.tight_layout()

    savedir = Path('figures/raw_eeg')
    savedir.mkdir(parents=True, exist_ok=True)
    fig.savefig(savedir/f'{session.ppt_id}_raw_eeg.png')
    fig.savefig(savedir/f'{session.ppt_id}_raw_eeg.svg')

def plot_average_trial_per_class(session, savepath):
    color_dict = {'rest': 'black',
                  'left': 'blue',
                  'right': 'orange'}

    n_channels = session.eeg.shape[1]
    n_rows = 20

    idc_per_col = np.array_split(np.arange(n_channels), n_channels/n_rows)
    max_rows = max([len(idc) for idc in idc_per_col])

    fig, ax = plt.subplots(nrows=max_rows, ncols=len(idc_per_col),
                           dpi=200, figsize=(24, 16))

    y_lab = session.trial_names
    y_num = session.trial_nums

    for label in np.unique(y_lab):

        if label == 'rest':  # TODO: Change to 'rest'
            # Trials start with left, so ends of trials first
            trial_borders = np.hstack((0, np.diff(y_num)))
            starts = np.where(trial_borders < 0)[0]
            ends = np.where(trial_borders > 0)[0]
            trials = [session.eeg[e:s, :] for s, e, in zip(starts, ends)]

        else:
            label_idc = np.where(y_lab==label)[0]
            trials = [session.eeg[y_num==trial_num, :] \
                     for trial_num in np.unique(session.trial_nums[label_idc])]
        
        min_size = min([tr.shape[0] for tr in trials])
        trials = np.dstack([tr[:min_size, :] for tr in trials]).mean(axis=2)
        
        for col_i, ax_col in enumerate(idc_per_col):
            for row_i, ch_num in enumerate(ax_col):
                ax[row_i, col_i].plot(trials[:, ch_num], linewidth=1, 
                                      color=color_dict[label], label=label if (col_i==0 and row_i==0) else None)
                ax[row_i, col_i].spines[['top', 'right', 'left', 'bottom']].set_visible(False)
                ax[row_i, col_i].tick_params(axis='x', which='both', 
                                                bottom=False, labelbottom=False)
                ax[row_i, col_i].tick_params(axis='y', which='both',
                                                left=False, labelleft=False)
                ax[row_i, col_i].set_ylabel(session.channels[ch_num])
                ax[row_i, col_i].set_ylim(-200, 200)
        
    fig.suptitle('Raw signal averaged per trial')
    fig.legend()

    savedir = Path('figures/raw_eeg_averaged_per_trial')
    savedir.mkdir(parents=True, exist_ok=True)
    fig.savefig(savedir/f'{session.ppt_id}_raw_eeg_averaged_per_trial.png')
    fig.savefig(savedir/f'{session.ppt_id}_raw_eeg_averaged_per_trial.svg')

def plot_task_correlation(session, savepath):
    eeg = session.eeg
    labels = session.trial_names

    classes = np.unique(labels)
    n_classes = len(classes)
    labels = np.vstack([labels==class_ for class_ in classes]).T
    cm = np.corrcoef(np.hstack((eeg, labels)), rowvar=False)
    
    task_correlation = cm[:-n_classes, -n_classes:]

    fig, ax = plt.subplots(nrows=1, ncols=1,
                           dpi=200)
    im = ax.imshow(task_correlation.T, vmin=-1, vmax=1)
    fig.savefig(f'{savepath}/raw_task_correlation.png')

def make_all(session, savepath):
    # raw_eeg(session, savepath)
    plot_average_trial_per_class(session, savepath)
    # plot_task_correlation(session, savepath)