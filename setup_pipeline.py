# Builtin
import logging
import io
import pstats
from pathlib import Path
from datetime import datetime as dt

# Local
from libs.load.load_yaml import load_yaml

c = load_yaml('./config.yml')

log_level_map = {'debug': logging.DEBUG,
                 'info': logging.INFO,
                 'warning': logging.WARNING,
                 'error': logging.ERROR}

def profiler(pr, savepath):
    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s)
    stats.dump_stats(f'{savepath}/profile.prof')

    with open(f'{savepath}/profile.txt', 'w') as f:
        ps = pstats.Stats(f'{savepath}/profile.prof', stream=f)
        ps.sort_stats(pstats.SortKey.TIME)
        ps.print_stats()

def setup_logger(path):

    log_filename = f'output.log'
    logging.basicConfig(format="[%(filename)10s:%(lineno)3s - %(funcName)20s()] %(message)s",
                        level=log_level_map.get(c.log.level, logging.INFO),
                        handlers=[
                            logging.FileHandler(path/f'{log_filename}'),
                            logging.StreamHandler()])

def debug():
    pass

def setup():
    main_path = Path(c.paths.save)
    today = dt.today().strftime('%Y%m%d_%H%M')
    path = main_path/today
    path.mkdir(parents=True, exist_ok=True)
    
    setup_logger(path)

    return path

