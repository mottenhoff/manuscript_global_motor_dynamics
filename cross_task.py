# Builtin
import cProfile
import logging
from pathlib import Path

import numpy as np

# Local
import setup
from libs.load.load_yaml import load_yaml
from libs.load import load_session

from libs.process_session import process_session as process
from libs import split_to_trials
from libs.learn_cross_task import decode_cross

np.random.seed(6227836)

logger = logging.getLogger(__name__)
c = load_yaml('./config.yml')

TASK = 0
FILTERS = 1
LOCATIONS = 0
PPT = 1

GRASP = 0
IMAGINE = 1
LOCATIONS = 2

def get_files(path: Path, ppt):
    return [path/ppt/'grasp.xdf',
            path/ppt/'imagine.xdf',
            path/ppt/'electrode_locations.csv']


def make_paths(savepath, filters, ppt_id):
    filter_str = ''.join(filters.keys())
    path = savepath/filter_str

    (path/ppt_id).mkdir(parents=True, exist_ok=True)

    logger.info(f'Running task: {filters} - {ppt_id}')

    return path

def setup_debug(session):
    session.eeg =         session.eeg[:c.debug.size, :]
    session.ts =          session.ts[:c.debug.size]
    session.trial_nums =  session.trial_nums[:c.debug.size]
    session.trial_names = session.trial_names[:c.debug.size]

    return session

def run_pipeline(savepath):

    tasks = ('grasp', 'imagine')
    filters = ({'beta': [12, 30]}, 
               {'hg':   [55, 90]},
               {'beta': [12, 30],
                'hg':   [55, 90]})

    ppts = c.ppts_to_include
    path_data = Path(c.paths.data)

    for filter_ in filters:
        
        for ppt in ppts:

            fileset = get_files(path_data, ppt)
            fullpath = make_paths(savepath, filter_, fileset[GRASP].parent.name)

            sessions = []

            for task in [fileset[GRASP], fileset[IMAGINE]]:

                files = [task, fileset[LOCATIONS]]

                session = load_session.load(files)
                session = process(session, filter_)
                session = split_to_trials.c2t(session)

                sessions += [session]

            decode_cross(sessions, fullpath)

def main():

    savepath = setup.setup()

    with cProfile.Profile() as pr:
        run_pipeline(savepath)

    setup.profiler(pr, savepath)

if __name__=='__main__':
    main()
