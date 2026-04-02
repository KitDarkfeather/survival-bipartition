from sapient.models.model import Model
from skopt.space import Categorical, Integer, Real
from sksurv.ensemble import RandomSurvivalForest as RSF

class RandomSurvivalForest(Model):
    """
    Generate a random survival forest model (as an ensemble of survival trees).
    """

    # hyperparameters
    trees=500
    maximum_tree_depth = 5
    minimum_node_size = 2
    minimum_leaf_size = 3
    node_random_feature_proportion = 0.25
    # node_random_feature_counts = ['sqrt', 'log2', None, 5, 10, 15, 20]
    # node_random_feature_count = 0
    max_samples = 1.0

    # hyperparameter space
    space = [
        # Integer(10, 3000, name='trees'),
        # Categorical([2, 3], name='maximum_tree_depth'),
        # Integer(4, 200, name='minimum_node_size'),
        Integer(1, 5, name='minimum_leaf_size'), # Integer(2, 100, name='minimum_leaf_size'),

        # this will fail if the feature count is less than 20
        Real(0.05, 1.0, prior='uniform', name='node_random_feature_proportion'),
        # Categorical(list(range(len(node_random_feature_counts))), name='node_random_feature_count'),

        # this will fail if the instance count is less than 100
        Real(0.01, 1.0, prior='uniform', name='max_samples'),
    ]

    def __init__(self, training='sa_training_subset', training_value=1, partition_frame=None, partition=None,
                 partition_value=None, bootstrap=0, bootstrap_df=None, feature_subset=None, verbose=True):
        self.name = 'random survival forest'
        self.nickname = 'rsf'
        super().__init__(training, training_value, partition_frame, partition, partition_value, bootstrap, bootstrap_df,
                         feature_subset, verbose)

    def generate(self, fold=1, verbose=True):
        if verbose:
            print(f'Generating a {self.name} model for fold {fold} ...')
        fold -= 1
        self.models[fold] = RSF(

            # hyperparameters
            n_estimators=self.trees,
            max_depth=self.maximum_tree_depth,
            min_samples_split=self.minimum_node_size,
            min_samples_leaf=self.minimum_leaf_size,
            max_features=self.node_random_feature_proportion,
            # max_features=self.node_random_feature_counts[self.node_random_feature_count],
            max_samples=self.max_samples,

            # parameters
            oob_score=True,
            n_jobs=-1,
            random_state=self.random_state,
            # verbose=2,
        )
        self.models[fold].fit(self.cv_train_sets[fold]['x'], self.cv_train_sets[fold]['y'])

if __name__ == '__main__':
    rsf = RandomSurvivalForest()
    # rsf.hpo()
    rsf.generate_cv()
    result = rsf.evaluate_cv()
    print(f'{rsf.name} results:')
    print(f'    Train IBS: {result["train integrated Brier score"]:.4f}')
    print(f'    Test IBS: {result["test integrated Brier score"]:.4f}')

    # rsf = RandomSurvivalForest(partition_frame='ICARE', partition='_icare_disjunction', partition_value=1)
    # rsf.generate_cv()
    # rsf.evaluate_cv()
    # rsf = RandomSurvivalForest(partition_frame='ICARE', partition='_icare_disjunction', partition_value=0)
    # rsf.generate_cv()
    # rsf.evaluate_cv()
