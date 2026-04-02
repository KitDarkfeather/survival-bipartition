import json
import numpy as np
import pandas as pd
from sksurv.nonparametric import kaplan_meier_estimator
import time
from umberto.datasets.dataset import Dataset

class ELSA(Dataset):
    """
    Merge ELSA raw files into a survival analysis dataset.

    [THIS CLASS IS A STUB.
    THE FULL CLASS WILL BE MADE AVAILABLE UPON REQUEST, AFTER PERMISSION TO ACCESS ELSA DATA IS OBTAINED]
    """

    def __init__(self, verbose=True):
        super().__init__(verbose=verbose)
        # print(f'DEBUG: The elsa data are located in "{self.shelf_path}"')
        self.features = {}
        self.df_features = None
        self.load_features()
        self.position = None
        self.source = None
        self.missing = None
        self.missing_dict = {
            'name': [],
            'description': [],
            'missing': [],
            'category': [],
            'subcategory': [],
        }
        self.orders = {}
        np.random.seed(42)

        # constants
        if (self.library_path / f'{self.folder}/json/cutoff.json').exists():
            with open(self.library_path / f'{self.folder}/json/cutoff.json', 'r') as json_file:
                self.calculated_cutoff = json.load(json_file)['cutoff']
        else:
            if self.verbose:
                print('The cutoff has not yet been calculated.')

    def load_features(self):
        """
        Loads feature data from a parquet file, processes it into a dictionary format, and
        stores it in the corresponding instance attributes.

        This method checks for the existence of a feature parquet file within the specified
        path. If the file exists, it loads the file into a DataFrame, organises its contents
        to create a dictionary of feature metadata, and assigns this dictionary along with
        the DataFrame to the instance's attributes.

        :raises FileNotFoundError: If the file is not found at the specified path.

        :return: None
        """
        if self.verbose:
            print('Loading features...')
        if (self.library_path / f'{self.folder}/parquet/features.parquet').exists():
            df = pd.read_parquet(self.library_path / f'{self.folder}/parquet/features.parquet', engine='pyarrow')
            self.features = dict(zip(df['name'], df[[
                'description',
                'category',
                'subcategory',
                'level',
                'source']].to_dict('records')))
            self.df_features = df

    def load(self, tables=None, tab=True, rows=None, silent=False):
        """
        Loads tables from a local file system or cached parquet files. The method determines the existence
        of requested tables in raw or cached file formats, loads the tables, optionally filters rows for testing
        purposes, and caches them as parquet files for faster future access. If no tables are specified, it will
        attempt to load all available raw tables.

        :param tables: List of table names to load. If None, all available raw tables will be loaded.
                       Can be a single string name or an iterable of strings.
        :type tables: Optional[Union[str, Iterable[str]]]
        :param tab: Flag indicating whether to load tab-delimited files ('tab') or Stata (.dta) files.
                    Defaults to True (load 'tab' files).
        :type tab: bool
        :param rows: An optional integer specifying the number of rows to load from each raw table.
                     If None, all rows will be loaded. This is useful for testing with smaller data.
        :type rows: Optional[int]
        :param silent: Boolean flag to suppress verbose output when set to True. Defaults to False (output enabled).
        :type silent: bool
        :return: The loaded tables as pandas DataFrame(s). If one table is loaded, a single DataFrame is returned.
                 If multiple tables are loaded, a list of DataFrames is returned. If no tables are loaded,
                 returns None.
        :rtype: Optional[Union[pd.DataFrame, List[pd.DataFrame]]]
        """
        if not silent:
            print(f'The ELSA data are located in "{self.shelf_path}"')
        raw = 'tab' if tab else 'dta'

        # determine what tables are available to load
        all_raw_tables = set([filename.stem for filename in (self.shelf_path / raw).iterdir()
                              if filename.is_file()])
        all_parquet_tables = set([filename.stem for filename in (self.shelf_path / 'parquet').iterdir()
                                  if filename.is_file()])

        # refine the request
        if tables is None:
            tables = all_raw_tables
        if isinstance(tables, str):
            tables = [tables]
        tables = sorted(list(tables))
        if not silent:
            print(f'Request to load the following tables: '
                  f'{', '.join([self.metadata[table]['name'] for table in tables])}')

        # compare the requested tables to the available tables and report problems
        for table in tables:
            if table not in all_raw_tables | all_parquet_tables:
                if not silent:
                    print(f'    Table "{self.metadata[table]['name']}" does not exist.')

        # divide the tables into raw and parquet sets
        tables = set(tables)
        parquet_tables = all_parquet_tables & tables
        raw_tables = (all_raw_tables & tables) - parquet_tables

        # load the requested raw tables
        for table in sorted(list(raw_tables)):
            if not silent:
                print(f'Loading table "{self.metadata[table]['name']}" from {raw} file...')

            # optional row restriction (for testing)
            more = {}
            if rows is not None:
                more['nrows'] = rows

            # read the raw file
            st = time.time()
            if not tab:
                try:
                    df = pd.read_stata(f'{self.shelf_path}/{raw}/{table}.{raw}')
                except ValueError:
                    df = pd.read_stata(f'{self.shelf_path}/{raw}/{table}.{raw}',
                                       convert_categoricals=False)
            else:
                df = pd.read_csv(f'{self.shelf_path}/{raw}/{table}.{raw}', sep='\t',
                                 lineterminator=self.line_terminator, comment=self.comment_delimiter,
                                 header=0, low_memory=False, **more)
            et = time.time()
            if not silent:
                print(f'    Raw load execution time: {et - st} seconds')

            # catalog dataframe for this instance
            self.dfs[table] = df

            # save the dataframe as a parquet file for faster loading in future runs
            if not silent:
                print(f'Saving dataframe "{table}" as parquet file...')
            st = time.time()
            self.save(df, table)
            et = time.time()
            if not silent:
                print(f'    parquet save execution time: {et - st} seconds')

        # load the requested parquet tables
        for table in sorted(list(parquet_tables)):

            # check to see if the table is already loaded
            if table in self.dfs:
                if not silent:
                    print(f'Table "{self.metadata[table]['name']}" is already loaded.')
            else:

                # read the parquet file
                if not silent:
                    print(f'Loading table "{self.metadata[table]['name']}" from parquet file...')
                st = time.time()
                df = pd.read_parquet(f'{self.shelf_path}/parquet/{table}.parquet', engine='pyarrow')
                et = time.time()
                if not silent:
                    print(f'    parquet load execution time: {et - st} seconds')

                # catalog dataframe for this instance
                self.dfs[table] = df
        dfs = list(self.dfs.values())
        return dfs if len(dfs) > 1 else (dfs[0] if len(dfs) > 0 else None)

    def verify(self):
        # stub
        pass

    def build(self):
        # stub
        pass

    def _check(self, df, feature):
        # stub
        self._frequency_table(df, feature)

    def _build_targets(self):
        # stub
        pass

    def _build_features(self):
        # stub
        pass

if __name__ == '__main__':
    elsa = ELSA()
