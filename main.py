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
from libs.learn import decode_dropout
from libs.plotting import plot_raw, plot_pre

logger = logging.getLogger(__name__)
c = load_yaml('./config.yml')

TASK = 0
FILTERS = 1

def is_valid(fileset):
    ppt_id = fileset[0].parts[1]

    if ppt_id not in c.ppts_to_include:
        return False

    if fileset[0].parts[-2] == '2':
        return False
    
    return True

def make_paths(savepath, task, filters, ppt_id):
    filter_str = ''.join(filters.keys())
    path = savepath/task/filter_str

    (path/ppt_id).mkdir(parents=True, exist_ok=True)

    logger.info(f'Running task: {task} - {filters} - {ppt_id}')

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

    for run_parameters in product(tasks, filters):        

        task = run_parameters[TASK]
        filters = run_parameters[FILTERS]

        filesets = get_filesets(c.paths.data, task)
    
        for fileset in filesets:

            if not is_valid(fileset):
                continue

            session = load_session.load(fileset)

            fullpath = make_paths(savepath, task, filters, session.kh_id)

            session = setup_debug(session) if c.debug.do else session

            if c.plot.raw: plot_raw.make_all(session, fullpath)

            session = process(session, filters)
            
            if session == 'invalid':
                logger.warning('Encountered invalid timestamps. Skipping current dataset')  
                continue

            session = split_to_trials.c2t(session)

            if c.plot.pre:    plot_pre.make_all(session, fullpath)
            if c.explore.do:  explore(session, fullpath)

            decode_dropout(session, fullpath)


def main():

    savepath = setup.setup()

    with cProfile.Profile() as pr:
        run_pipeline(savepath)

    setup.profiler(pr, savepath)

if __name__=='__main__':
    main()
