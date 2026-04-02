import numpy as np
from numpy.lib import recfunctions as rfn
import pandas as pd
from sapient.analytics import Analytics
from sapient.plots.brier import Brier
from sksurv.metrics import concordance_index_censored, brier_score, integrated_brier_score
from sksurv.preprocessing import OneHotEncoder
import statistics as stat
from tabulate import tabulate
import xgboost as xgb
from skopt import gp_minimize, dummy_minimize
from skopt.callbacks import DeltaYStopper, EarlyStopper
from skopt.space import Real, Integer
from skopt.plots import plot_convergence, plot_evaluations, plot_gaussian_process, plot_objective

class Model(Analytics):
    """
    Generate a survival analysis model.
    """

    # seed
    random_state = 42

    # hyperparameters
    types = []

    def __init__(self, training='sa_training_subset', training_value=1, partition_frame=None, partition=None,
                 partition_value=None, bootstrap=0, bootstrap_df=None, feature_subset=None, verbose=True, folds=5):
        """
        Initialises an instance of the class to handle the creation and processing of modelling datasets for a
        machine learning workflow. The class facilitates cross-validation through fold-based dataset partitions,
        one-hot encoding of categorical features, and management of ensemble models.

        :param training: Name or identifier of the training set. Defaults to `'sa_training_subset'`.
        :param training_value: Numeric value associated with the training subset. Defaults to `1`.
        :param partition_frame: A DataFrame used to manage data partitioning. Can be None.
        :param partition: Specific partition to use for training/testing. Can be None.
        :param partition_value: Numeric value to identify a specific partition for training/testing. Can be None.
        :param bootstrap: An integer indicating whether bootstrap sampling is enabled (1) or disabled (0). Defaults to `0`.
        :param bootstrap_df: A DataFrame to hold bootstrap data. Can be None.
        :param feature_subset: A list or subset of feature names to be used in the model. Can be None.
        :param verbose: A boolean flag to enable or disable verbose output during initialisation. Defaults to `True`.
        :param folds: An integer indicating the number of folds for cross-validation. Defaults to `5`.
        """
        super().__init__(training, training_value, partition_frame, partition, partition_value, bootstrap, bootstrap_df,
                         feature_subset, verbose)
        self.ensemble = []
        self.ensemble_frequencies = dict([(m.__name__, 0) for m in self.types])

        # initialize model attributes
        self.verbose = verbose
        self.folds = folds
        self.instances = self.df.shape[0]
        self.models = [None] * folds

        # generic name (overwritten by subclasses)
        if not hasattr(self, 'name'):
            self.name = 'Model'
        if not hasattr(self, 'nickname'):
            self.nickname = 'mod'

        # create modelling datasets
        try:
            self.xt = OneHotEncoder().fit_transform(self.df[self.nominal_features].astype('category'))
            self.df = pd.concat([self.df, self.xt], axis=1)
        except ValueError:
            pass
        self.structured_events = self._create_fold_sets(self.df)['y']
        self.cv_test_sets = []
        self.cv_train_sets = []
        for fold in range(folds):
            test = self.df[(20 + fold * 16 < self.df['sa_cv_centile'])
                           & (self.df['sa_cv_centile'] <= 20 + (fold + 1) * 16)].copy()
            self.cv_test_sets.append(self._create_fold_sets(test))
            train = self.df[((20 + fold * 16 >= self.df['sa_cv_centile']) & (self.df['sa_cv_centile'] > 20))
                            | (self.df['sa_cv_centile'] > 20 + (fold + 1) * 16)].copy()
            self.cv_train_sets.append(self._create_fold_sets(train))
        if self.verbose:
            print(f'The train datasets contain [{', '.join([str(len(self.cv_train_sets[fold]['y'])) 
                                                            for fold in range(folds)])}] instances.')
            print(f'The test datasets contain [{', '.join([str(len(self.cv_test_sets[fold]['y'])) 
                                                           for fold in range(folds)])}] instances.')
        self.test_instances = sum([len(self.cv_train_sets[fold]['y']) for fold in range(folds)])

    def _create_fold_sets(self, frame):

        # convert boolean event into binary event list
        event = frame['event'].map({1: True, 0: False}).to_list()

        # convert time + 1 to a list
        time = frame['time'].to_list()
        time = [t + 1.0 for t in time]

        # convert time lower bound + 1 to a list
        time_lower_bound = (frame['time_lower_bound'] + 1).to_list()

        # convert time upper bound + 1 to a list
        frame['time_upper_bound'] += 1
        frame.loc[frame['event'] == False, 'time_upper_bound'] = +np.inf
        time_upper_bound = frame['time_upper_bound'].to_list()

        # create a structured array of time and event
        y = np.array(list(zip(event, time)), dtype=[('event', '?'), ('time', '<f8')])

        # create a data frame of just features
        try:
            x = frame[self.numeric_features_plugged + self.numeric_features_floating + self.boolean_features
                      + self.xt.columns.tolist()]
        except AttributeError:

            # no categorical features
            x = frame[self.numeric_features_plugged + self.numeric_features_floating + self.boolean_features]

        # return a dictionary of features and targets for models
        return {
            'time_lower_bound': time_lower_bound,
            'time_upper_bound': time_upper_bound,
            'y': y,
            'x': x,
        }

    @staticmethod
    def _summary(df, limit=None):
        summ = pd.DataFrame(df.dtypes, columns=['Data type'])
        summ['Total count'] = df.shape[0]
        summ['Missing count'] = df.isnull().sum().values
        summ['Missing percent'] = df.isnull().sum().values / len(df) * 100
        summ['Unique count'] = df.nunique().values
        summ['Duplicate count'] = summ['Total count'] - summ['Unique count']
        desc = pd.DataFrame(df.describe(include='all').transpose())
        summ['Minimum'] = desc['min'].values
        summ['Maximum'] = desc['max'].values
        if limit is not None:
            summ = summ.iloc[:limit, :]
        return tabulate(summ, headers='keys', tablefmt='fancy_grid')

    def hpo(self, calls=80, initial=15, random=False, verbose=True):
        """
        Using scikit-optimise to perform hyperparameter optimisation.
        """

        # initialise iteration counter
        iteration = 0

        # objective function
        def objective(parameters):
            for p, n in zip(parameters, [item.name for item in self.space]):
                setattr(self, n, p)
            # print(f'DEBUG: Check parameters: {parameters}')
            save = self.verbose
            self.verbose = False
            try:
                self.generate_cv()
                objective_result = self.evaluate_cv()
                ibs = objective_result['train integrated Brier score']
            except ValueError:

                # during HPO, some parameter combinations can yield invalid models, so penalise them
                ibs = 1.0
            self.verbose = save
            # print(f'DEBUG: Check IBS: {ibs}')
            return ibs

        def callback(progress):
            nonlocal iteration
            iteration += 1
            print(f'Iteration {iteration}: parameters: {progress.x}, objective: {progress.fun}')

        # perform the Bayesian optimisation or random search
        if random:
            print(f'\nRandom search:')
            result = dummy_minimize(
                func=objective,
                dimensions=self.space,
                n_calls=calls,
                random_state=42,
                callback=[
                    callback,
                ],
            )
        else:
            print(f'\nBayesian optimisation:')
            result = gp_minimize(
                func=objective,
                dimensions=self.space,
                n_calls=calls,
                n_initial_points=initial,

                # options are 'EI', 'PI', 'LCB', and 'gp_hedge'
                acq_func='EI',
                random_state=42,
                callback=[
                    callback,
                    _NoImprovementStopper(),
                    # DeltaYStopper(delta=1e-6, n_best=10),
                ],
            )

        # save the results
        for parameter, name in zip(result.x, [item.name for item in self.space]):
            setattr(self, name, parameter)

        # show the results
        print(f'Best score: {result.fun}')
        print(f'Best parameters:')
        for parameter, name in zip(result.x, [item.name for item in self.space]):
            if type(parameter) in [np.bool_, bool]:
                parameter = bool(parameter)
            else:
                try:
                    if int(parameter) == parameter:
                        parameter = int(parameter)
                    else:
                        parameter = round(parameter, 3)
                except TypeError:
                    pass
            print(f'\t{name}: {parameter}')
        plt.close('all')
        plt.figure(num=0, clear=True)
        plot_convergence(result)
        has_models = bool(getattr(result, 'models', []))
        if len(self.space) > 1 and has_models:
            plot_objective(result)
            plot_evaluations(result)
        plt.show()
        return result.x

    def generate(self, fold=1, verbose=True):
        raise NotImplementedError

    def generate_cv(self):
        if self.verbose:
            print(f'Generating {self.name} models for {self.folds} cross-validation folds ...')
        for fold in range(1, self.folds + 1):
            self.generate(fold, verbose=False)

    def _get_estimate(self, f_index, cv_sets):
        unstructured_y = rfn.structured_to_unstructured(cv_sets[f_index]['y'])
        if self.nickname == 'aft':
            x = cv_sets[f_index]['x'].values.astype(np.float32)
            pred = self.models[f_index].predict(xgb.DMatrix(x))

            # use a stable risk score for concordance metrics.
            # for AFT, larger predicted (log-)time => lower risk, so negate it.
            estimate = -np.asarray(pred, dtype=np.float64)
        else:
            pred = self.models[f_index].predict(cv_sets[f_index]['x'])
            estimate = np.asarray(pred, dtype=np.float64)

        # hard fail early with a helpful message rather than letting metrics explode later
        if not np.all(np.isfinite(estimate)):
            bad = np.where(~np.isfinite(estimate))[0][:10]
            raise ValueError(
                f'Non-finite values in estimate for fold index {f_index}: '
                f'{estimate[bad]} (showing up to 10). '
                f'Check model predictions and input features.'
            )
        return unstructured_y, estimate

    def _get_predictions(self, f_index, cv_sets, times):
        survival_curves = self.models[f_index].predict_survival_function(cv_sets[f_index]['x'])
        predictions = np.asarray([[fn(t) for t in times] for fn in survival_curves])
        return predictions

    def evaluate(self, fold=1):
        if self.verbose:
            print(f'Evaluating {self.name} model on fold {fold} ...')
        f_index = fold - 1

        # concordance
        unstructured_y_test, test_estimate = self._get_estimate(f_index, self.cv_test_sets)
        unstructured_y_train, train_estimate = self._get_estimate(f_index, self.cv_train_sets)
        test_harrell_df = pd.DataFrame({
            'event_indicator': unstructured_y_test[:, 0].astype(np.bool),
            'event_time': unstructured_y_test[:, 1],
            'estimate': test_estimate
        })
        c_index_train = concordance_index_censored(
            unstructured_y_train[:, 0].astype(bool),
            unstructured_y_train[:, 1],
            train_estimate
        )[0]
        if self.verbose:
            print(f'    The C-index for the train data for fold {fold} is {c_index_train}')
        try:
            c_index_oob = self.models[f_index].oob_score_
            if self.verbose:
                print(f'    The OOB C-index for the train data for fold {fold} is {c_index_oob}')
        except AttributeError:
            c_index_oob = None
        c_index_test = concordance_index_censored(
            unstructured_y_test[:, 0].astype(bool),
            unstructured_y_test[:, 1],
            test_estimate
        )[0]
        if self.verbose:
            print(f'    The C-index for the test data fold {fold} is {c_index_test}')

        # calculate time points for Brier scores
        times = np.linspace(1, 87.5, 100)  # Time points from 0 to 86.5 months

        # calculate predictions
        train_predictions = self._get_predictions(f_index, self.cv_train_sets, times)
        test_predictions = self._get_predictions(f_index, self.cv_test_sets, times)

        # calculate Brier scores for the test data
        brier_scores = []
        train_ibs = integrated_brier_score(self.cv_train_sets[f_index]['y'], self.cv_train_sets[f_index]['y'],
                                           train_predictions, times)
        if self.verbose:
            print(f'    The integrated Brier score for the train data for fold {fold} is {train_ibs}')
        test_ibs = integrated_brier_score(self.cv_train_sets[f_index]['y'], self.cv_test_sets[f_index]['y'],
                                          test_predictions, times)
        if self.verbose:
            print(f'    The integrated Brier score for the test data for fold {fold} is {test_ibs}')
        survivals = self.models[f_index].predict_survival_function(self.cv_test_sets[f_index]['x'])
        for t in times:

            # calculate Brier score at time t
            predictions = [fn(t) for fn in survivals]
            _, score = brier_score(self.cv_train_sets[f_index]['y'], self.cv_test_sets[f_index]['y'], predictions, t)
            brier_scores.append(score[0])

        # create a dataFrame for the Brier scores
        brier_df = pd.DataFrame({'x': times, 'y': brier_scores})

        # create the Brier data directory if it doesn't exist
        brier = Brier()
        brier.data_path.mkdir(parents=True, exist_ok=True)

        # save the Brier scores to a CSV file
        brier_df.to_csv(brier.data_path / f'{self.nickname}_brier_scores_fold_{fold}.csv', index=False)

        # create the Brier score plot
        # brier.df = brier_df
        # brier.plot(f'{self.nickname}_brier_scores_fold_{fold}', x_label='Time in months', y_label='Brier Score',
        #           x_domain=[0, 86.5], y_range=[0.0, brier_df['y'].max() * 1.1], size=22)
        # if self.verbose:
        #     print(f'    Brier score plot fold {fold} has been created and saved.')

        # finish up
        return {
            'train c index': c_index_train,
            'OOB c index': c_index_oob,
            'test Harrell c index data': test_harrell_df,
            'test c index': c_index_test,
            'train integrated Brier score': train_ibs,
            'test integrated Brier score': test_ibs,
            'Brier score over time': brier_df,
        }

    def evaluate_cv(self):
        if self.verbose:
            print(f'Evaluating cross-validated {self.name} model ...')

        # get evaluation metrics for each fold
        metrics = [self.evaluate(fold) for fold in range(1, self.folds + 1)]

        # calculate the average train set concordance
        c_index_train = stat.mean([metric['train c index'] for metric in metrics])
        if self.verbose:
            print(f'    The cross-validated average C-index for the train data folds is {c_index_train:.3f}')

        # attempt to calculate the average OOB train set concordance
        try:
            c_index_oob = stat.mean([metric['OOB c index'] for metric in metrics])
            if self.verbose:
                print(f'    The cross-validated average OOB C-index for the train data folds is {c_index_oob:.3f}')
        except TypeError:
            c_index_oob = None

        # calculate the actual test set concordance
        test_harrell_df = metrics[0]['test Harrell c index data'].copy()
        for fold in range(1, self.folds):
            test_harrell_df = pd.concat([test_harrell_df, metrics[fold]['test Harrell c index data'].copy()])
        c_index_actual_test = concordance_index_censored(test_harrell_df['event_indicator'],
                                                         test_harrell_df['event_time'], test_harrell_df['estimate'])[0]
        if self.verbose:
            print(f'    The cross-validated actual C-index for the test data folds is {c_index_actual_test:.3f}')

        # calculate the average test set concordance
        c_index_test = stat.mean([metric['test c index'] for metric in metrics])
        if self.verbose:
            print(f'    The cross-validated average C-index for the test data folds is {c_index_test:.3f}')

        # calculate the average train set IBS
        train_ibs = stat.mean([metric['train integrated Brier score'] for metric in metrics])
        if self.verbose:
            print(f'    The cross-validated average integrated Brier score for the train data folds is {train_ibs:.3f}')

        # calculate the average test set IBS
        test_ibs = stat.mean([metric['test integrated Brier score'] for metric in metrics])
        if self.verbose:
            print(f'    The cross-validated average integrated Brier score for the test data folds is {test_ibs:.3f}')

        # average cross-validated test set Brier scores over time
        brier_df = metrics[0]['Brier score over time']
        for fold in range(1, self.folds):
            brier_df['y'] += metrics[fold]['Brier score over time']['y']
        brier_df['y'] /= self.folds
        brier = Brier()

        # save the Brier scores to a CSV file
        brier_df.to_csv(brier.data_path / f'{self.nickname}_brier_scores.csv', index=False)

        # create the Brier score plot
        brier.df = brier_df
        brier.plot(f'{self.nickname}_brier_scores', x_label='Time in months', y_label='Brier Score',
                  x_domain=[0, 86.5], y_range=[0.0, brier_df['y'].max() * 1.1], size=22)
        if self.verbose:
            print(f'    Brier score plot has been created and saved.')
        return {
            'train c index': c_index_train,
            'OOB c index': c_index_oob,
            'test Harrell c index data': test_harrell_df,
            'test c index': c_index_test,
            'train integrated Brier score': train_ibs,
            'test integrated Brier score': test_ibs,
            'Brier score over time': brier_df,
        }

class _NoImprovementStopper(EarlyStopper):
    def __init__(self, k=10, threshold=1e-12, min_calls=0):
        self.k = k
        self.threshold = abs(threshold)
        self.min_calls = int(min_calls)

    def _criterion(self, result):
        vals = np.array(result.func_vals, dtype=float)
        return_value = None

        # never stop before we have enough evaluations to be meaningful (and to allow surrogate fitting).
        if len(vals) < self.min_calls:
            return_value = False

        # need at least k + 1 points to compare 'best up to -k - 1' vs 'best up to now'
        elif len(vals) <= self.k:
            return_value = False

        # normal check of no improvement in last k evaluations
        else:
            cum_min = np.minimum.accumulate(vals)
            return_value = (cum_min[-self.k - 1] - cum_min[-1]) <= self.threshold
        return return_value

if __name__ == '__main__':
    m = Model()


