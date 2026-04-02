import numpy as np
from sapient.models.partitioned_model import PartitionedModel
from time import time

class IBSDifferenceTest(object):

    # bootstrap iterations
    bootstrap_iterations = 1_000

    def __init__(self, model, hyperparameters, partition, tolerance=1e-6):
        start = time()
        pm = PartitionedModel(model, hyperparameters, partition, hpo=False, verbose=False)
        result = pm.results[-1]
        observed_delta = result['ibs: test: combined'] - result['ibs: test: all']
        print(f'Observed IBS difference: {observed_delta:.3f} '
              f'({-observed_delta / result['ibs: test: all'] * 100.0:.1f}%)')
        print(observed_delta)

        # bootstrap the distribution of IBS differences
        print(f'Bootstrapping {self.bootstrap_iterations} IBS differences ...')
        bootstrap_differences = []
        last_p = 0.0
        iteration = 1
        while iteration < self.bootstrap_iterations:
            try:
                pm = PartitionedModel(model, hyperparameters, partition, ibs=True, hpo=False, verbose=False)
                result = pm.results[-1]
                bootstrap_delta = result['ibs: test: combined'] - result['ibs: test: all']
                # print(f'[{iteration}] Bootstrap IBS difference: {bootstrap_delta:.3f} '
                #       f'({-bootstrap_delta / result['ibs: test: all'] * 100.0:.1f}%)')
                bootstrap_differences.append(bootstrap_delta)
                bootstrap_differences_array = np.array(bootstrap_differences)

                # two-sided p-value: proportion of bootstrap differences more extreme than observed difference
                p_value1 = np.mean(np.abs(bootstrap_differences_array) >= np.abs(observed_delta))
                p_value2 = np.mean(bootstrap_differences_array >= 0.0)
                p_value3 = np.mean(bootstrap_differences_array - np.mean(bootstrap_differences_array)
                                   >= np.abs(observed_delta))
                print(f'[{iteration}] Bootstrap IBS difference: {bootstrap_delta:.3f} '
                      f'({-bootstrap_delta / result['ibs: test: all'] * 100.0:.1f}%), '
                      f'mean: {np.mean(bootstrap_differences_array):.6f}, '
                      f'std: {np.std(bootstrap_differences_array):.6f}, '
                      + ('' if iteration == 1
                      else f'distance: {-observed_delta / np.std(bootstrap_differences_array):.3f}, ')
                      + f'p-values: {p_value1:.3f} {p_value2:.3f} {p_value3:.3f}')
                # print(f'[{iteration}] Two-sided p-values: {p_value1:.3f} {p_value2:.3f}')
                if iteration > 200:
                    if abs(p_value1 - last_p) < tolerance:
                        break
                    last_p = p_value1
            except ValueError:
                print(f'Bootstrap failed for iteration {iteration}. Retrying.')
                iteration -= 1
            iteration += 1

        # final reporting
        bootstrap_differences_array = np.array(bootstrap_differences)

        # two-sided p-value: proportion of bootstrap differences more extreme than observed difference
        p_value1 = np.mean(np.abs(bootstrap_differences_array) >= np.abs(observed_delta))
        p_value2 = np.mean(bootstrap_differences_array >= 0.0)
        p_value3 = np.mean(bootstrap_differences_array - np.mean(bootstrap_differences_array)
                           >= np.abs(observed_delta))
        print(f'[{iteration}] p-values: {p_value1:.3f} {p_value2:.3f} {p_value3:.3f}')
        print(p_value1)
        print(p_value2)
        print(p_value2)
        print(observed_delta)
        print(list(bootstrap_differences_array))
        elapsed = (time() - start) / 60.0 / 60.0
        print(f'Time elapsed: {elapsed:.2f} hours')

if __name__ == '__main__':
    from sapient.models.cox_elastic_net import CoxElasticNet
    from sapient.models.random_survival_forest import RandomSurvivalForest
    from sapient.models.accelerated_failure_time import AcceleratedFailureTime

    # generate full results for 4 partitioning methods and 3 modelling methods
    for _partition in [
        'ICARE',
        # 'OC1-u',
        # 'OC1-c',
        # 'S-PCA'
    ]:
        for _model in [
            (CoxElasticNet, {'l1_ratio': 0.01, 'normalize': True}),
            # (AcceleratedFailureTime, {}),
            # (RandomSurvivalForest, {})
        ]:
            IBSDifferenceTest(_model[0], _model[1], _partition)
