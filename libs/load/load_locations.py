import csv

def load(path):

    with open(path) as f:
        reader = csv.DictReader(f)
        data = {row['electrode_name_1']: row['location'] for row in reader}

    return data

if __name__=='__main__':
    from pathlib import Path

    load(Path(r'./data/kh13/electrode_locations.csv'))