import pandas as pd
from SurvSet.data import SurvLoader
from umberto.datasets.dataset import Dataset

class Survset(Dataset):

    def __init__(self):
        super().__init__()
        # print(f'DEBUG: The Survival Set data are located in "{self.shelf_path}"')
        self.loader = SurvLoader()

    def update_metadata(self):

        # get Survival Set's metadata
        df_ss = self.loader.df_ds

        # get metadata metadata
        df = self.dfs['metadata']
        keep = list(df.columns)

        # do a temporary merge
        df = df.merge(right=df_ss, how='left', left_on='filename', right_on='ds')

        # update features and instances
        df['features'] = df['n_fac'] + df['n_num']
        df['instances'] = df['n']

        # save metadata dataset
        self.save(df[keep], 'metadata')

        # reload metadata
        self.load_metadata()

    def get(self, filename):
        print(f'Loading the "{self.metadata[filename]['name']}" dataset...')
        if (self.shelf_path / f'parquet/{filename}.parquet').exists():
            print(f'    Loading "{filename}" table from parquet file...')
            df = pd.read_parquet(self.shelf_path / f'parquet/{filename}.parquet', engine='pyarrow')
        else:
            print(f'    Loading "{filename}" table from SurvSet...')
            df, _ = self.loader.load_dataset(ds_name=filename).values()
            print(f'    Saving "{filename}" dataframe as parquet file...')
            self.save(df, filename)
        self.dfs[filename] = df
        return df

    # alias for get method
    def load(self, filename):
        return self.get(filename)

if __name__ == '__main__':
    ss = Survset()
    # ss.update_metadata()
    # for name in ['burn', 'colon', 'cost', 'csl']:
    #     ss.load(name)
    #     print(ss.dfs[name].dtypes)
