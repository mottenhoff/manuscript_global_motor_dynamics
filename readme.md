# Ottenhoff et al. - Global motor dynamics - Invariant neural representations of motor behavior in distributed brain-wide recordings.
Maarten C. Ottenhoff*, Maxime Verwoert, Sophocles Goulis , Louis Wagner , Johannes P. van Dijk , Pieter L. Kubben, and Christian Herff \
*Corresponding author

## Description
Code accompanying the paper. The data accompanying the paper will be made available for free on publication.

## Installation
Using a conda environment: \
`conda create -n <your_env_name> python=3.9.7`\
`conda activate <your_env_name>`\
`pip install -r requirements.txt`

## To run the code to recreate results:
1. `python main.py`\
in `./results`, a datetime folder will be created holding all the results. After `main.py` ran sucessfully, rename the datetime folder to `full_run`. Then the subsequent scripts have access to the correct data.

2. `python cross_task.py`
3. `python transfer_learning`
4. `python make_figures.py`

