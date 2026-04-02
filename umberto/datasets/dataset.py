import calendar
import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

class Dataset(object):
    library_path = Path(__file__).resolve().parent.parent / 'library'

    @classmethod
    def _tau(cls, month, year):
        return year * 12 + (month - 1)

    @classmethod
    def _tau_inverse(cls, tau):
        return f'{calendar.month_name[tau % 12 + 1]} {tau // 12}'

    @classmethod
    def _tau_inverse_sortable(cls, tau):
        return tau // 12 + (tau % 12 + 1) / 100

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.folder = self.__class__.__name__.lower()
        self.dfs = {}
        self.metadata = {}
        self.shelf_path = self.library_path / self.folder
        if self.shelf_path.exists():
            self.load_metadata()

    def load_metadata(self, refresh=False):

        # check if metadata exists in parquet
        if not (self.library_path / f'{self.folder}/parquet/metadata.parquet').exists() or refresh:

            # if not, load metadata into dataframe and persist to parquet file
            if self.verbose:
                print('Loading metadata table from csv file...')
            df = pd.read_csv(self.library_path / f'{self.folder}/csv/metadata.csv', header=0, low_memory=False)
            if self.verbose:
                print(f'Saving metadata dataframe as parquet file...')
            self.save(df,'metadata' )

        # load metadata.parquet into a dataframe
        df = pd.read_parquet(self.library_path / f'{self.folder}/parquet/metadata.parquet', engine='pyarrow')

        # instantiate a dictionary with the metadata keyed by file name
        self.metadata = dict(zip(df['filename'],
                                 df[['name', 'features', 'instances', 'description']].to_dict('records')))

    def save(self, dataframe, name):
        # noinspection PyArgumentList
        arrow_table = pa.Table.from_pandas(df=dataframe, preserve_index=True)
        pq.write_table(arrow_table, f'{self.library_path}/{self.folder}/parquet/{name}.parquet')

    @staticmethod
    def _frequency_table(df, name):
        counts = df.value_counts(name)
        sum_of_counts = 0
        print(f'\n{name.upper()} FREQUENCIES')
        for value in sorted(counts.index):
            print(f'{value:<20}{counts[value]}')
            sum_of_counts += counts[value]
        print(f'TOTAL: {sum_of_counts:,.0f}')

if __name__ == '__main__':
    ds = Dataset()
