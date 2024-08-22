import matplotlib.cm as cm

def kh_to_ppt():
    return {'kh9':  'p1', 
            'kh10': 'p2', 
            'kh11': 'p3', 
            'kh12': 'p4', 
            'kh13': 'p5', 
            'kh15': 'p6', 
            'kh18': 'p7', 
            'kh30': 'p8'}

def color_map():
    cmap = cm.tab10
    # cmap = cm.Dark2
    ppts = kh_to_ppt().values()
    return dict(zip(ppts, [cmap(i) for i in range(len(ppts))]))