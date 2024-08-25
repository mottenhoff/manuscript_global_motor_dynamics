from itertools import product, combinations_with_replacement
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from libs import mappings

TASKS   = ('grasp', 'imagine')
FILTERS = ('beta', 'hg', 'betahg')
PCS = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
N_TASKS = len(TASKS)
N_FILTERS = len(FILTERS)
N_PCS = len(PCS)

SAMPLES = 1
TRIAL_DURATION = 3
TARGET_FS = 1024

FIRST_COMPONENT = 0
SECOND_COMPONENT = 1
RELATIVE_VARIANCE = 1
ABSOLUTE_VARIANCE = 0

BOTTOM_LEFT = 3
COLOR_REST = cm.plasma(0)
COLOR_MOVE = cm.plasma(127)

@dataclass
class ProcessedData:
    task: str
    filter_: str
    ppt: str
    explained_variance: np.array
    principal_components: np.array
    labels: np.array

def load_data_ppt(path, task, filter_):
    path = path/task/filter_

    result = []
    for ppt in mappings.PPTS:

        labels = np.load(path/ppt/'labels.npy')
        labels = np.where(labels=='rest', 'rest', 'move')

        result.append(
            ProcessedData(
                task,
                filter_,
                ppt,
                np.load(path/ppt/'full_explained_variance.npy'),
                np.load(path/ppt/'processed_data.npy'),
                labels
            )
        )

    return result

def load_data(path, tasks, filters):

    data = defaultdict(dict)
    for task, filter_ in product(tasks, filters):

        data[task][filter_] = load_data_ppt(path, task, filter_)

    return data

def extract_move_rest_per_condition(data):
    '''Data is list of ProcessedData instances'''

    move, rest = [], []
    for ppt in data:
        
        ppt_pcs = ppt.principal_components
        
        # One participant has double the TARGET_FS so
        # that needs to be reduced to be able to stack them
        # + 100 is a safety to prevent reducing data with a
        # a few samples more than TARGET_FS * TRIAL_DURATION
        decimate = 2 if ppt_pcs.shape[SAMPLES] > TARGET_FS * TRIAL_DURATION +100 else 1

        ppt_pcs = ppt_pcs[:, 0:decimate*TARGET_FS*TRIAL_DURATION:decimate, :]

        ppt_move = ppt_pcs[ppt.labels=='move', :, :].mean(axis=0)
        ppt_rest = ppt_pcs[ppt.labels=='rest', :, :].mean(axis=0)
        
        move.append(ppt_move)
        rest.append(ppt_rest)
    
    return np.stack(move), np.stack(rest)

def plot_explained_variances(data):

    savepath_scree = Path('./figures/scree_indices.txt')
    if savepath_scree.exists():
        savepath_scree.unlink()

    fig, axs = plt.subplots(nrows=N_TASKS,
                            ncols=N_FILTERS,
                            figsize=(12, 8))
    axs = axs.flatten()
    cmap = mappings.color_map()
    x = np.arange(max(PCS))
    x_ticks = np.arange(0, max(PCS)+1, 10)
    y_ticks = np.arange(0, 101, 20) / 100
    y_tick_labels = np.arange(0, 101, 20)

    task_filter_combinations = list(product(TASKS, FILTERS))
    for idx, (task, filter_) in enumerate(task_filter_combinations):
        
        # Plot individual lines
        for ppt in data[task][filter_]:
            explained_variance = ppt.explained_variance
            cumulative_ev = np.cumsum(explained_variance[:, RELATIVE_VARIANCE])

            axs[idx].step(x, 
                          cumulative_ev, 
                          alpha=0.5,
                          color=cmap[ppt.ppt])

        # Plot averages
        explained_variance = np.stack([ppt.explained_variance for ppt in data[task][filter_]])
        cumulative_ev = np.cumsum(explained_variance, axis=1)
        mean_cumulative_ev = cumulative_ev[:, :, RELATIVE_VARIANCE].mean(axis=0)
        std_cumulative_ev = cumulative_ev[:, :, RELATIVE_VARIANCE].std(axis=0)
        
        scree_line = np.where(explained_variance[:, :, ABSOLUTE_VARIANCE].mean(axis=0)<1)[0][0]


        axs[idx].plot(x, mean_cumulative_ev, color='black')
        axs[idx].fill_between(x, 
                              mean_cumulative_ev - std_cumulative_ev,
                              mean_cumulative_ev + std_cumulative_ev,
                              color='black',
                              alpha=0.2)
        axs[idx].axvline(scree_line, ymin=0, ymax=1, color='black', linestyle='--')

        axs[idx].set_xticks(x_ticks)
        axs[idx].set_yticks(y_ticks)
        axs[idx].set_yticklabels(y_tick_labels)

        axs[idx].spines[['top', 'right']].set_visible(False)

        if idx == BOTTOM_LEFT:
            axs[idx].set_xlabel('Principal components', fontsize='x-large')
            axs[idx].set_ylabel('Cumulative\n Explained Variance [%]', fontsize='x-large')
        
        # Save for labels in final figures
        with open('./figures/scree_indices.txt', 'a+') as f:
            f.write(f'{task}_{filter_}: {scree_line}\n')

    fig.tight_layout()
    fig.savefig(f'./figures/figure_5_explained_variance.png')
    fig.savefig(f'./figures/figure_5_explained_variance.svg')
    plt.close('all')

def plot_first_n_components_per_ppt(data, n_components=10, name=None):

    savepath = Path('./figures/pcs/')
    savepath.mkdir(exist_ok=True, parents=True)

    task_filter_combinations = list(product(TASKS, FILTERS))

    for idx, (task, filter_) in enumerate(task_filter_combinations):

        # --- Get data --- 
        all_move, all_rest = extract_move_rest_per_condition(data[task][filter_])

        for ppt in range(mappings.N_PPTS):
            
            move, rest = all_move[ppt, :, :], all_rest[ppt, :, :]

            fig, axs = plt.subplots(nrows=n_components,
                                    ncols=n_components,
                                    figsize=(10, 10))
            for ax in axs.flatten(): 
                ax.axis('off')
            
            rest -= rest[0, :]
            move -= move[0, :]

            for y_idx, x_idx in combinations_with_replacement(np.arange(n_components), 2):

                axs[x_idx, y_idx].axis('on')
                axs[x_idx, y_idx].spines[['top', 'right']].set_visible(False)
                axs[x_idx, y_idx].set_xticks([])
                axs[x_idx, y_idx].set_yticks([])

                if y_idx == 0:
                    axs[x_idx, y_idx].set_ylabel(f'PC {x_idx+1}', fontsize='x-large')

                if x_idx == n_components-1:
                    axs[x_idx, y_idx].set_xlabel(f'PC {y_idx+1}', fontsize='x-large')

                if x_idx == y_idx:
                    axs[x_idx, y_idx].plot(rest[:, x_idx], color=COLOR_REST, linewidth=1)
                    axs[x_idx, y_idx].plot(move[:, x_idx], color=COLOR_MOVE, linewidth=1)
                    continue
                
                axs[x_idx, y_idx].plot(rest[:, x_idx], rest[:, y_idx], color=COLOR_REST, linewidth=1)
                axs[x_idx, y_idx].plot(move[:, x_idx], move[:, y_idx], color=COLOR_MOVE, linewidth=1)

            fig.suptitle(f'{task.capitalize()} {filter_.capitalize()} | Participant {idx+1}', fontsize=30)
            fig.tight_layout()

            fig.savefig(savepath/f'figure_5_pcs_{task}_{filter_}_ppt{idx}.png')
            fig.savefig(savepath/f'figure_5_pcs_{task}_{filter_}_ppt{idx}.svg')
            plt.close('all')
    
def plot_figure_component_space(data, individual_trajectories=False):

    fig, axs = plt.subplots(nrows=N_TASKS, ncols=N_FILTERS, figsize=(12, 8))
    axs = axs.flatten()

    task_filter_combinations = list(product(TASKS, FILTERS))

    for idx, (task, filter_) in enumerate(task_filter_combinations):

        # --- Get data --- 
        move, rest = extract_move_rest_per_condition(data[task][filter_])

        # Center at 0
        move -= move[:, :1, :]  # :1 instead if 0 to keep dimensions
        rest -= rest[:, :1, :]

        # --- Plot it ---
        if individual_trajectories:
            for ppt_idx in range(mappings.N_PPTS):
                axs[idx].plot(move[ppt_idx, :, FIRST_COMPONENT],
                              move[ppt_idx, :, SECOND_COMPONENT],
                              linewidth=-.1,
                              color=COLOR_MOVE)
                axs[idx].plot(rest[ppt_idx, :, FIRST_COMPONENT ],
                              rest[ppt_idx, :, SECOND_COMPONENT],
                              linewidth=.1,
                              color=COLOR_REST)

        move_mean, rest_mean = move.mean(axis=0), rest.mean(axis=0)

        axs[idx].plot(move_mean[:, FIRST_COMPONENT ],
                      move_mean[:, SECOND_COMPONENT],
                      linewidth=2,
                      color=COLOR_MOVE,
                      label='Move')
        axs[idx].plot(rest_mean[:, FIRST_COMPONENT ],
                      rest_mean[:, SECOND_COMPONENT],
                      linewidth=2,
                      color=COLOR_REST,
                      label='Rest')
        
        axs[idx].spines[['top', 'right']].set_visible(False)
        
        xticks = [-1, 0, 1]
        axs[idx].set_xticks(xticks)
        axs[idx].set_yticks(xticks)
        axs[idx].set_xticklabels(xticks, fontsize='x-large')
        axs[idx].set_yticklabels(xticks, fontsize='x-large')

    axs[0].set_ylabel('Execute\n\n', fontsize=22)
    axs[3].set_ylabel('Imagine\n\nPC 2', fontsize=22)
    axs[3].set_xlabel('PC 1', fontsize='xx-large')

    axs[0].set_title('Beta', fontsize=22)
    axs[1].set_title('High-gamma', fontsize=22)
    axs[2].set_title('Both', fontsize=22)

    axs[2].legend(bbox_to_anchor=(1.1, 1), frameon=False, fontsize='xx-large')

    fig.tight_layout()

    if individual_trajectories:
        fig.savefig('./figures/figure_5_principal_component_space_individual_trajectories.png')
        fig.savefig('./figures/figure_5_principal_component_space_individual_trajectories.svg')
    else:
        fig.savefig('./figures/figure_5_principal_component_space.png')
        fig.savefig('./figures/figure_5_principal_component_space.svg')

    plt.close('all')

def make(path: Path):

    data = load_data(path, TASKS, FILTERS)

    plot_figure_component_space(data)
    plot_figure_component_space(data, individual_trajectories=True)
    plot_first_n_components_per_ppt(data)
    plot_explained_variances(data)


