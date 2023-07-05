from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
# import tensortools as tt

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

    plt.savefig(f'{session.kh_id}_test.png')

    return

def principal_component_analysis(session, savepath):


    plot_pca_over_all_data(session, savepath)

    eeg = (session.eeg - session.eeg.mean(axis=0)) / session.eeg.std(axis=0)

    pca = PCA().fit(eeg)
    pcs = pca.transform(eeg)



    return 

def make_all(session, savepath):

    # tensor_component_analysis(session, savepath)
    principal_component_analysis(session, savepath)
    
    pass