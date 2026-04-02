from sapient.models.model import Model
from skopt.space import Categorical, Real
from sksurv.linear_model import CoxnetSurvivalAnalysis as CEN

class CoxElasticNet(Model):
    """
    Generate a Cox elastic net model.
    """

    # parameters
    max_iter = 100_000
    tol = 1e-7
    fit_baseline_model = True

    # hyperparameters
    normalize = False
    l1_ratio = 0.5

    # hyperparameter space
    space = [
        # Categorical([True, False], name='normalize'),
        Real(1e-6, 1.0, prior='uniform', name='l1_ratio')
    ]

    def __init__(self, training='sa_training_subset', training_value=1, partition_frame=None, partition=None,
                 partition_value=None, bootstrap=0, bootstrap_df=None, feature_subset=None, verbose=True):
        self.name = 'Cox elastic net'
        self.nickname = 'cen'
        super().__init__(training, training_value, partition_frame, partition, partition_value, bootstrap, bootstrap_df,
                         feature_subset, verbose)

    def generate(self, fold=1, verbose=True):
        if verbose:
            print(f'Generating a {self.name} model for fold {fold} ...')
        fold -= 1
        self.models[fold] = CEN(

            # hyperparameters
            normalize=self.normalize,
            l1_ratio=self.l1_ratio,

            # parameters
            max_iter=self.max_iter,
            tol=self.tol,
            fit_baseline_model=self.fit_baseline_model,
        )
        self.models[fold].fit(self.cv_train_sets[fold]['x'], self.cv_train_sets[fold]['y'])

if __name__ == '__main__':
    cen = CoxElasticNet()
    # cen.hpo()
    cen.generate_cv()
    result = cen.evaluate_cv()
    print(f'{cen.name} results:')
    print(f'    Train IBS: {result["train integrated Brier score"]:.4f}')
    print(f'    Test IBS: {result["test integrated Brier score"]:.4f}')

    # cen = CoxElasticNet(partition_frame='ICARE', partition='_icare_disjunction', partition_value=1)
    # cen.l1_ratio = 0.01
    # cen.generate_cv()
    # cen.evaluate_cv()
    # cen = CoxElasticNet(partition_frame='ICARE', partition='_icare_disjunction', partition_value=0)
    # cen.l1_ratio = 0.01
    # cen.generate_cv()
    # cen.evaluate_cv()
