import logging
from pathlib import Path
from itertools import product

def get_filenames(parent, extension, keywords=[], exclude=[]):
    main = Path(parent)

    if not main.exists():
        logging.error(f'Cannot access path <{main}>')
        raise NameError

    ext = extension.strip('.')
    
    files = []
    for file in main.rglob(f'*.{ext}'):
        
        if any(kw in file.name for kw in exclude):
            continue

        if not keywords:
            files.append(file)
            continue

        if any(kw in file.name for kw in keywords):
            files.append(file)
    
    return files

def get_filesets(parent, keywords, exclude=[]):
    keywords = [keywords] if type(keywords) is not list else keywords
    exclude =   [exclude] if type(exclude)  is not list else exclude

    parent = Path(parent)

    seeg_filenames =    get_filenames(parent, 'xdf', keywords=keywords, exclude=exclude)
    contact_filenames = get_filenames(parent, 'csv', keywords=['electrode_locations'])

    possible_sets = product(seeg_filenames, contact_filenames)

    return [(s, l) for s, l in possible_sets if s.parts[1] == l.parts[1]]


if __name__=='__main__':
    paths = get_filesets('./data', 'grasp', 'imagine')

    print()