from pathlib import Path
from umberto.datasets.elsa import ELSA
from umberto.datasets.survset import Survset

class Datatables(object):
    """
    Indifferent access to library datasets
    """
    library_path = Path(__file__).resolve().parent.parent / 'library'

    def __init__(self):
        self.metadata = {}

    def get(self, ds_name):
        for source in self.library_path.iterdir():
            ds = None
            if source.stem == 'survset':
                ds = Survset()
            elif source.stem == 'ELSA':
                ds = ELSA()
            if ds_name in ds.metadata:
                df = ds.load(ds_name)
                self.metadata = ds.metadata[ds_name]
                break
        else:
            raise Exception(f'The dataset "{ds_name}" is not in the library.')
        return df

if __name__ == '__main__':
    Datatables()