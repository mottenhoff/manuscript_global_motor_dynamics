import matplotlib.cm as cm

N_PPTS = 8
PPTS = [f'p{i+1}' for i in range(N_PPTS)]

def color_map(cmap=cm.tab10):
    return dict(zip(PPTS, [cmap(i) for i, _ in enumerate(PPTS)]))