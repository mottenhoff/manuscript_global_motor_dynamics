# Ottenhoff et al. - Global motor dynamics - Invariant neural representations of motor behavior in distributed brain-wide recordings.
Maarten C. Ottenhoff*, Maxime Verwoert, Sophocles Goulis , Louis Wagner , Johannes P. van Dijk , Pieter L. Kubben, and Christian Herff \
*Corresponding author

# Abstract
*Objective*: Motor-related neural activity is more widespread than previously thought, as pervasive brain-wide neural correlates
of motor behavior have been reported throughout the animal brain. It is unknown whether these global patterns exist in
humans as well, and to what extent. *Approach*: Here, we use a decoding approach to capture and characterize brain-wide
neural correlates of movement. We recorded invasive electrophysiological data from stereotactic electroencephalographic
electrodes implanted in eight epilepsy patients who performed an executed and imagined grasping task. Combined, these
electrodes cover the whole brain, including deeper structures such as the hippocampus, insula and basal ganglia. We extract
a low-dimensional representation and classify move from rest using a Riemannian decoder. *Main results*: We reveal global
neural dynamics that are predictive across tasks and across participants. Using an ablation analysis, we demonstrate that
these dynamics remain remarkably stable under loss of information. Similarly, the dynamics remain stable across participants,
as we were able to predict movement across participants using transfer learning. *Significance*: Our results show that decodable
global motor-related neural dynamics within a low-dimensional space exist. The dynamics are predictive of movement, near
brain-wide and present in all our participants. The results broaden the scope to brain-wide investigates, and may allow to
combine datasets of multiple participants with varying electrode locations or calibrationless neural decoder.

# Related resources

Data repository: `<coming soon>`

## Installation
Using a conda environment: \
`conda create -n <your_env_name> python=3.9.7`\
`conda activate <your_env_name>`\
`pip install -r requirements.txt`

## To run the code to recreate results:

1. Download the data from `Related resources` and save in a new folder named `./Data`

2. `python main.py`\
in `./results`, a datetime folder will be created holding all the results. After `main.py` ran sucessfully, rename the datetime folder to `full_run`so that `make_figures.py` has access to the right folders.
3. `python cross_task.py`\
Rename folder in results to `cross_task`
4. `python transfer_learning`
5. `python make_figures.py`