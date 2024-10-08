from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mne.filter import filter_data
from matplotlib.gridspec import GridSpec
from pyriemann.estimation import Covariances

from libs.load import load_session

COLORS = {'rest':  cm.plasma(0),
          'left':  cm.plasma(127),
          'right': cm.plasma(255),
          'move':  cm.plasma(127)}

NORM = 'symlog'

def annotate_imshow(ax, values):

    for x in np.arange(values.shape[0]):
        for y in np.arange(values.shape[1]):
            
            color = 'grey' if x == y else 'lightgrey'
            text = ax.text(y, x, f'{values[x, y]:.2f}', 
                           ha='center', va='center', 
                           color=color, fontsize='x-large')

    return ax

def load_raw_data(path):
    return load_session.load([path/'grasp.xdf', path/'electrode_locations.csv'])

def plot_raw_data(ax, session):
    fig, ax = plt.subplots()

    selection = (20000, 15)  # sample x channels

    session.trial_names = np.where(session.trial_names=='rest', 'rest', 'move')

    for i in range(selection[1]):
        ax.plot(session.eeg[:selection[0], i+15] + i*1000, color='black', linewidth=1)

    # xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for class_ in set(session.trial_names):

        idc = np.where(session.trial_names[:selection[0]]==class_, 1, 0)
        start, end = np.where(np.diff(idc) > 0)[0], np.where(np.diff(idc) < 0)[0]

        if class_=='move':
            start = np.hstack((0, start))
            end =   np.hstack((end, selection[0]))

        for s, e in zip(start, end):
            ax.axvspan(s, e, ylim[0], ylim[1], alpha=0.5, color=COLORS[class_], zorder=1.9)
        
        xpos = (end[0] - start[0]) / 2 + start[0]
        ax.annotate(class_.capitalize(), (xpos, ylim[1]), (xpos, ylim[1]*1.03),
                    ha='center', color=COLORS[class_], fontsize='xx-large')

    ax.axis('off')

    ax.set_xlim(0, selection[0])

    fig.savefig('./figures/figure_0_raw_data.svg')
    fig.savefig('./figures/figure_0_raw_data.png')

    return ax

def plot_sample_covariance_matrix(ax):
    fig, ax = plt.subplots()
    example_cov = np.load('./results/example_covs/example_trial.npy')
    example_cov = example_cov.squeeze()

    # npcs = example_cov.shape[0]

    # im = ax.imshow(example_cov, cmap='plasma', norm=NORM)
    ax = annotate_imshow(ax, example_cov)

    # ax.set_xticks(np.arange(5))
    # ax.set_yticks(np.arange(5))
    # ax.set_xticklabels([f'PC{i}' for i in np.arange(1, npcs+1)])
    # ax.set_yticklabels([f'PC{i}' for i in np.arange(1, npcs+1)])

    # ax.set_title('Trial covariance matrix', fontsize='x-large')
    ax.axis('off')
    fig.savefig('./figures/figure_0_example_cov.svg')
    fig.savefig('./figures/figure_0_example_cov.png')

    return ax

def plot_rest_geometric_mean(ax):
    fig, ax = plt.subplots()

    cov = np.load('./results/example_covs/rest_mean_cov.npy')

    ax.imshow(cov, cmap='plasma', norm=NORM)
    ax = annotate_imshow(ax, cov)

    # Annotate right
    # ax.text(5, 2, "Move", ha="left", va="center", fontsize='xx-large')

    ax.axis('off')

    fig.savefig('./figures/figure_0_rest_cov_mean.svg')
    fig.savefig('./figures/figure_0_rest_cov_mean.png')

    return ax

def plot_move_geometric_mean(ax):
    fig, ax = plt.subplots()
    
    cov = np.load('./results/example_covs/move_mean_cov.npy')
    
    ax.imshow(cov, cmap='plasma', norm=NORM)
    ax = annotate_imshow(ax, cov)

    # ax.set_title('Geometric mean', fontsize='xx-large')

    # ax.text(5, 2, "Rest", ha="left", va="center", fontsize='xx-large')

    ax.axis('off')
        
    fig.savefig('./figures/figure_0_move_cov_mean.svg')
    fig.savefig('./figures/figure_0_move_cov_mean.png')

    return ax

def covmat_to_str(cov):
    row_strings = [f'{row[0]:>6.01f}{row[1]:>6.01f}{row[2]:>6.01f}\n'
                   for row in cov]

    return ''.join(row_strings).strip()

def plot_3d_lowdimrep_2class():
    lowpass = 1
    # annot = 0

    for example_trial in [18]:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
       
        # Load
        xyz = np.load('results/full_run/grasp/beta/kh18/processed_data.npy')
        labels = np.load('results/full_run/grasp/beta/kh18/labels.npy')
        labels = np.where(labels=='rest', 'rest', 'move')

        for class_ in set(labels):
        
            pcs = xyz[np.where(labels==class_), :, :3].squeeze()
        
            # pcs = pcs[example_trial, :, :]
            pcs = pcs.mean(axis=0)
            pcs -= pcs[0, :]

            cov = Covariances().transform(pcs[np.newaxis, : ,:].transpose(0, 2, 1))
            print(class_, cov)
        
            if lowpass:
                pcs = filter_data(pcs.T, 1024, None, 0.05).T

            ax.scatter(*pcs.T, label=class_, marker='o', s=2, color=COLORS[class_])

            ax.quiver(pcs[-2, 0], pcs[-2, 1], pcs[-2, 2],
                      pcs[-1, 0]-pcs[-2, 0],
                      pcs[-1, 1]-pcs[-2, 1],
                      pcs[-1, 2]-pcs[-2, 2],
                      arrow_length_ratio=1, 
                      linewidth=4, 
                      length=30, 
                      color=COLORS[class_])

            # Set axis line and background
            for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
                axis.set_pane_color((0.95, 0.95, 0.95, .5))
                axis.line.set_linewidth(3)

            # Remove lines and ticks
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            # Set view
            ax.view_init(azim=-155, elev=35)

        fig.savefig(f'figures/figure_0_plot3d_{example_trial}.png')
        fig.savefig(f'figures/figure_0_plot3d_{example_trial}.svg')

    return ax

def plot_3d_lowdimrep_1class():
    lowpass = 1
    annot = 0

    for example_trial in [1, 2, 3]:
       
        # Load
        xyz = np.load('results/full_run/grasp/beta/kh18/processed_data.npy')
        labels = np.load('results/full_run/grasp/beta/kh18/labels.npy')
        labels = np.where(labels=='rest', 'rest', 'move')

        for class_ in set(labels):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            
            pcs = xyz[np.where(labels==class_), :, :3].squeeze()
        
            pcs = pcs[example_trial, :, :]
            # pcs -= pcs[0, :]

            if lowpass:
                pcs = filter_data(pcs.T, 1024, None, 1).T

            ax.scatter(*pcs.T, label=class_, marker='o', s=2, color=COLORS[class_])

            ax.quiver(pcs[-2, 0], pcs[-2, 1], pcs[-2, 2],
                    pcs[-1, 0]-pcs[-2, 0],
                    pcs[-1, 1]-pcs[-2, 1],
                    pcs[-1, 2]-pcs[-2, 2],
                    arrow_length_ratio=1, linewidth=4, 
                    length=30, color=COLORS[class_])

            # Set axis line and background
            for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
                axis.set_pane_color((0.95, 0.95, 0.95, .5))
                axis.line.set_linewidth(2)

            # Remove lines and ticks
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            # Set view
            ax.view_init(azim=-155, elev=35)

            fig.savefig(f'figures/figure_0_plot3d_{class_}_{example_trial}.svg')
            fig.savefig(f'figures/figure_0_plot3d_{class_}_{example_trial}.png')
        # fig.close('all')

    return ax


def make(path):

    example_cov = np.load('./results/example_covs/example_trial.npy')
    rest = np.load('./results/example_covs/rest_mean_cov.npy')
    move = np.load('./results/example_covs/move_mean_cov.npy')
    d_rest = 0.789248
    d_move = 0.586734

    data_raw = load_raw_data(Path('./data/kh18'))

    grid = GridSpec(2, 4)
    fig = plt.figure(figsize=(16, 9))

    plot_raw_data(fig.add_subplot(grid[1, 0]), data_raw)
    plot_sample_covariance_matrix(fig.add_subplot(grid[1, 0]))
    plot_move_geometric_mean(fig.add_subplot(grid[1, 1]))
    plot_rest_geometric_mean(fig.add_subplot(grid[1, 3]))
    plot_3d_lowdimrep_1class()
    plot_3d_lowdimrep_2class()