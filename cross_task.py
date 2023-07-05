# Builtin
import cProfile
import logging
from itertools import product

# 3th party
import yaml

# Local
import setup
from libs.explore import explore
from libs.load.load_yaml import load_yaml
from libs.load import load_session
from libs.load.load_filenames import get_filesets

from libs.process_session import process_session as process
from libs import split_to_trials
from libs.learn_cross_task import decode as decode_cross_task

from libs.plotting import plot_raw, plot_pre

logger = logging.getLogger(__name__)
c = load_yaml('./config.yml')

TASK = 0
FILTERS = 1
LOCATIONS = 0
PPT = 1

def combine_filesets(filesets):
    file_dict = {}

    for fileset in filesets:
        ppt = fileset[0].parts[PPT]

        if ppt not in file_dict:
            file_dict.update({ppt: []})

        if fileset[0].parts[-2] == '2':
            continue

        file_dict[ppt] += fileset

    combined = [sorted(set(files)) for files in file_dict.values()]

    return combined

def is_valid(fileset):
    ppt_id = fileset[0].parts[1]

    if ppt_id not in c.ppts_to_include:
        return False

    if fileset[0].parts[-2] == '2':
        return False

    return True

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

    for filter_ in filters:

        filesets = get_filesets(c.paths.data, list(tasks))
        filesets = combine_filesets(filesets)

        for fileset in filesets:

            if not is_valid(fileset):
                continue

            fullpath = make_paths(savepath, filter_, fileset[0].parts[PPT])

            sessions = []
            if len(fileset[0].parts) == 4:   # in case of multiple sessions
                fileset = [fileset[-1], fileset[0], fileset[1]]

            assert 'grasp'   in str(fileset[1]) and \
                   'imagine' in str(fileset[2]) and \
                   fileset[0].suffix == '.csv',\
                   f'Wrong fileorder for {fileset}'

            for task in fileset[1:]:
                files = [task, fileset[LOCATIONS]]

                session = load_session.load(files)

                session = setup_debug(session) if c.debug.do else session

                session = process(session, filter_)
                
                if session == 'invalid':
                    logger.warning('Encountered invalid timestamps. Skipping current dataset')  
                    continue

                session = split_to_trials.c2t(session)

                sessions += [session]

            decode_cross_task(sessions, fullpath)

    print('done')

def main():

    savepath = setup.setup()

    with cProfile.Profile() as pr:
        run_pipeline(savepath)

    setup.profiler(pr, savepath)

if __name__=='__main__':
    main()
