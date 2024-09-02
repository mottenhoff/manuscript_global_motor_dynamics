from pathlib import Path

from libs.figures.figure_1 import make as make_fig_1
from libs.figures.figure_2 import make as make_fig_2
from libs.figures.figure_3 import make as make_fig_3
from libs.figures.figure_4 import make as make_fig_4
from libs.figures.figure_5 import make as make_fig_5

def make_all(path: Path):
    make_fig_1(path/'full_run')             # Schematics
    make_fig_2(path/'full_run')             # General results
    make_fig_3(path/'full_run')             # Dropout
    make_fig_4(Path('./transfer_results'))  # Transfer learning
    make_fig_5(path/'full_run')              # Principal components

if __name__=='__main__':
    path = Path(r'./results/')
    make_all(path)