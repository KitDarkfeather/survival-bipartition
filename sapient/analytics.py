from umberto.datasets.elsa import ELSA
from random import seed, randint, random

class Analytics(object):
    """
    Parent class of partition and modelling method classes.
    """

    def __init__(self, training='sa_training_subset', training_value=1, partition_frame=None, partition=None,
                 partition_value=None, bootstrap=0, bootstrap_df=None, feature_subset=None, verbose=True):
        """
        Initialises an object for processing the ELSA dataset and categorising features based
        on specific criteria.

        This constructor initialises the dataset, its features, and categorises them into
        numeric, nominal, and boolean features. It also handles multiple versions of features
        with missing values, retains analytic variables, and applies filtering based on
        training and partitioning parameters.

        :param training: Feature name used for training. The default is 'sa_training_subset'.
        :param training_value: An integer or indicator for a specific training subset. The default is 1.
        :param partition_frame: Optional data frame for partitioning data. The default is None.
        :param partition: Feature name of partitioning used on the data. The default is None.
        :param partition_value: An integer or indicator to identify the partition to use. The default is None.
        :param bootstrap: Determines whether bootstrapping is enabled during initialisation. The default is 0.
        :param bootstrap_df: Specifies an existing bootstrap data frame from a prior run. The default is None.
        :param feature_subset: A list of feature names to retain. The default is None, which retains all features.
        :param verbose: Print progress messages. The default is True.
        """

        # load ELSA dataset
        dataset = 'ELSA'
        self.elsa = ELSA(verbose=verbose)
        if verbose:
            print(f'LOADING THE {dataset} DATASET ...')
        self.elsa.load(dataset, silent=True)
        df = self.elsa.dfs[dataset].copy()
        if verbose:
            print(f'The {dataset} dataset contains {df.shape[0]:,} instances.')

        # subset train or test if specified
        df = df[df[training] == training_value].copy()
        if verbose:
            print(f'The {dataset} {'train' if training_value == 1 else 'test'} dataset '
                  f'contains {df.shape[0]:,} instances.')

        # keep the feature list
        self.features = self.elsa.features.copy()

        # add partition if specified
        if partition_frame is not None and partition is not None:
            self.elsa.load(f'{dataset}-{partition_frame}', silent=True)
            df['partition'] = self.elsa.dfs[f'{dataset}-{partition_frame}'][partition]
            if verbose:
                print(f'The {partition_frame} partition has been added to the dataset.')

        # apply the bootstrap and bootstrap sample size if specified
        if 0 < bootstrap <= 1:
            seed()
            df = df.sample(frac=bootstrap, replace=True,
                           random_state=randint(0, 100_000_000)).reset_index(drop=True)
            self.df_bootstrap = df.copy()
            if verbose:
                print(f'A new bootstrap sample has been applied to the dataset.')
        elif bootstrap == 2:
            df = bootstrap_df.copy()
            if verbose:
                print(f'An existing bootstrap sample has been applied to the dataset.')

        # subset partition if specified
        if partition_value is not None:
            df = df[df['partition'] == partition_value].copy()
            if verbose:
                print(f'The {dataset} {'train' if training_value == 1 else 'test'} dataset '
                      f'{'"imminent"' if partition_value == 1 else '"eventual"'} partition '
                      f'contains {df.shape[0]:,} instances.')

        # set random seed for reproducibility
        seed(42)

        # update the feature list with plugged, missing low, and missing high features
        all_columns = df.columns
        for column in list(self.features.keys()):
            if f'{column}_mv' in all_columns:
                for variation in [
                    ('pl', 'plugged'),
                    ('lo', 'low'),
                    ('hi', 'high'),
                ]:
                    self.features[f'{column}_{variation[0]}'] = self.features[column].copy()
                    self.features[f'{column}_{variation[0]}']['description'] += f' with MVs {variation[1]}'
                del self.features[column]

        # create quadratic age
        df['sa_age_squared'] = df['sa_age'] * df['sa_age']
        self.features['sa_age_squared'] = self.features['sa_age'].copy()
        self.features['sa_age_squared']['description'] += ' squared'

        # subset features if specified
        if feature_subset is not None:
            feature_list = [feature for feature in self.features
                            if self.features[feature]['category'] not in ['target', 'validation']]
            full_count = len(feature_list)
            if verbose:
                print(f'Original features: {full_count}')
            seed()
            if feature_subset == 'square root':
                cutoff = pow(full_count, 1 / 2) / full_count
            elif feature_subset == 'cube root':
                cutoff = pow(full_count, 1 / 3) / full_count
            else:
                cutoff = feature_subset / full_count
            for feature in feature_list:
                if random() > cutoff:
                    del self.features[feature]
            feature_list = [feature for feature in self.features
                            if self.features[feature]['category'] not in ['target', 'validation']]
            full_count = len(feature_list)
            if verbose:
                print(f'Final features: {full_count}')

        # retain analytic variables only, but with multiple versions of missing value features
        include = ['idauniq'] + list(self.features.keys())
        if partition_frame is not None and partition is not None:
            include.append('partition')
        self.df = df[include].copy()

        # create categorized feature lists
        self.numeric_features_plugged = [
            feature for feature in self.features
            if (self.features[feature]['level'] == 'ratio'
            and not feature.endswith('_hi')
            and not feature.endswith('_lo')
            and self.features[feature]['category'] not in ['target', 'validation'])
            or feature.endswith('_change')
        ]
        if verbose:
            print(f'There are {len(self.numeric_features_plugged)} numeric features in the dataset (missings plugged).')
        self.numeric_features_floating = [
            feature for feature in self.features
            if self.features[feature]['level'] == 'ratio'
            and (feature.endswith('_hi') or feature.endswith('_lo'))
            and self.features[feature]['category'] not in ['target', 'validation']
        ]
        if verbose:
            print(f'There are {len(self.numeric_features_floating)} numeric features with floating missing values.')
        self.nominal_features = [feature for feature in self.features
                                 if self.features[feature]['level'] in ['ordinal', 'categorical']
                                 and not feature.endswith('_change')
                                 and self.features[feature]['category'] not in ['target', 'validation']
                                 ]
        if verbose:
            print(f'There are {len(self.nominal_features)} nominal features in the dataset.')
        self.boolean_features = [feature for feature in self.features
                                 if self.features[feature]['level'] in ['boolean',]
                                 and self.features[feature]['category'] not in ['target', 'validation']
                                 ]
        if verbose:
            print(f'There are {len(self.boolean_features)} boolean features in the dataset.')


if __name__ == '__main__':
    Analytics()
    # Analytics(partition_frame='ICARE', partition='_icare_disjunction', partition_value=1)