from sapient.models.model import Model
import numpy as np
import pandas as pd
from scipy.stats import norm
from skopt.space import Categorical, Integer, Real
from sksurv.metrics import concordance_index_censored
from tabulate import tabulate
import warnings
import xgboost as xgb

class AcceleratedFailureTime(Model):
    """
    Generate an accelerated failure time model using XGBoost.

    This implementation uses XGBoost's survival:aft objective for accelerated failure time modeling
    and includes evaluation using Brier score over time.
    """

    # parameters
    objective = 'survival:aft'
    eval_metric = 'aft-nloglik'

    # hyperparameters
    learning_rate = 0.05
    minimum_split_loss = 0.0
    maximum_depth = 6
    minimum_child_weight = 1.0
    maximum_delta_step = 0.0
    subsample = 0.8
    sampling_methods = ['uniform', 'gradient_based']
    sampling_method = 0
    column_sample_by_tree = 1.0
    column_sample_by_level = 1.0
    column_sample_by_node = 1.0

    l2_lambda = 1.0
    l1_alpha = 0.0
    tree_methods = ['exact', 'approx', 'hist']
    tree_method = 2
    scale_positive_weight = 1.0
    # updater
    # refresh_leaf
    # process_type
    grow_policies = ['depthwise', 'lossguide']
    grow_policy = 0
    maximum_leaves = 0
    maximum_bins = 256
    parallel_tree_count = 1
    # monotone_constraints
    # interaction_constraints
    aft_loss_distributions = ['normal', 'logistic', 'extreme']
    aft_loss_distribution = 0
    aft_loss_distribution_alpha = 1.20

    # training hyperparameters
    number_of_boosting_rounds = 15

    # hyperparameter space
    _EPSILON = 1e-6
    _LOG_EPSILON = 1e-12
    space = [
        Real(0.0, 1.0, prior='uniform', name='learning_rate'), #
        Real(_LOG_EPSILON, 10.0, prior='log-uniform', name='minimum_split_loss'), #
        Integer(1, 10, name='maximum_depth'), #
        Real(_LOG_EPSILON, 10.0, prior='log-uniform', name='minimum_child_weight'), #
        # Real(_LOG_EPSILON, 10.0, prior='log-uniform', name='maximum_delta_step'),
        Real(_EPSILON, 1.0, prior='uniform', name='subsample'), #
        # Categorical(list(range(len(sampling_methods))), name='sampling_method'),
        Real(_EPSILON, 1.0, prior='uniform', name='column_sample_by_tree'), #
        Real(_EPSILON, 1.0, prior='uniform', name='column_sample_by_level'), #
        Real(_EPSILON, 1.0, prior='uniform', name='column_sample_by_node'), #
        Real(_LOG_EPSILON, 10.0, prior='log-uniform', name='l2_lambda'), #
        Real(_LOG_EPSILON, 10.0, prior='log-uniform', name='l1_alpha'), #
        # Categorical(list(range(len(tree_methods))), name='tree_method'),
        # Real(_LOG_EPSILON, 10.0, prior='log-uniform', name='scale_positive_weight'),
        # Categorical(list(range(len(grow_policies))), name='grow_policy'),
        # Integer(0, 10000, name='maximum_leaves'),
        # Integer(1, 10000, name='maximum_bins'),
        # Integer(1, 10000, name='parallel_tree_count'),
        # Categorical(list(range(len(aft_loss_distributions))), name='aft_loss_distribution'),
        # Real(_LOG_EPSILON, 10.0, prior='log-uniform', name='aft_loss_distribution_alpha'),
        Integer(5, 1000, name='number_of_boosting_rounds'), #
    ]

    def __init__(self, training='sa_training_subset', training_value=1, partition_frame=None, partition=None,
                 partition_value=None, bootstrap=0, bootstrap_df=None, feature_subset=None, verbose=True):
        self.name = 'accelerated failure time'
        self.nickname = 'aft'
        self.parameters = None
        self.feature_names = None
        super().__init__(training, training_value, partition_frame, partition, partition_value, bootstrap, bootstrap_df,
                         feature_subset, verbose)

    def generate(self, fold=1, verbose=True):
        """
        Generate and train the XGBoost AFT model.
        """
        if verbose:
            print(f'Generating a {self.name} model for fold {fold} ...')
        fold -= 1

        # set up XGBoost parameters
        self.parameters = {
            'eta': self.learning_rate,
            'gamma': self.minimum_split_loss,
            'max_depth': self.maximum_depth,
            'min_child_weight': self.minimum_child_weight,
            'max_delta_step': self.maximum_delta_step,
            'subsample': self.subsample,
            'sampling_method': self.sampling_methods[self.sampling_method],
            'colsample_bytree': self.column_sample_by_tree,
            'colsample_bylevel': self.column_sample_by_level,
            'colsample_bynode': self.column_sample_by_node,
            'reg_lambda': self.l2_lambda,
            'reg_alpha': self.l1_alpha,
            'tree_method': self.tree_methods[self.tree_method],
            'scale_pos_weight': self.scale_positive_weight,
            'grow_policy': self.grow_policies[self.grow_policy],
            'max_leaves': self.maximum_leaves,
            'max_bin': self.maximum_bins,
            'num_parallel_tree': self.parallel_tree_count,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'aft_loss_distribution': self.aft_loss_distributions[self.aft_loss_distribution],
            'aft_loss_distribution_scale': self.aft_loss_distribution_alpha,
            'seed': self.random_state,
        }

        # address parameter conflicts
        tree_method = self.parameters['tree_method']
        colsample_bynode = self.parameters['colsample_bynode']
        if tree_method == "exact" and float(colsample_bynode) != 1.0:
            if verbose:
                print(
                    'Warning: tree_method = \'exact\' is incompatible with colsample_bynode != 1. '
                    'Switching tree_method to \'hist\' to avoid XGBoostError.'
                )
            self.parameters['tree_method'] = 'hist'

        # create DMatrix with proper data types
        d_train = xgb.DMatrix(self.cv_train_sets[fold]['x'].values.astype(np.float32))
        d_train.set_float_info('label_lower_bound', np.array(self.cv_train_sets[fold]['time_lower_bound']))
        d_train.set_float_info('label_upper_bound', np.array(self.cv_train_sets[fold]['time_upper_bound']))

        # train the model
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.models[fold] = xgb.train(self.parameters, d_train, num_boost_round=self.number_of_boosting_rounds)

        # store feature names for later use
        self.models[fold].feature_names_in_ = self.cv_train_sets[fold]['x'].columns.tolist()

        # add functionality to the model attribute to duck-type it for compatibility with scikit-survival
        self.models[fold].score = self.score
        self.models[fold].predict_survival_function = self.predict_survival_function
        self.models[fold].predict_risk = self.predict

        # make the current fold model callable from score and predict_survival_function without passing fold
        self.model = self.models[fold]

        # store feature names for later use
        self.feature_names = self.cv_train_sets[fold]['x'].columns.tolist()

    def predict_survival_function(self, x):

        # determine underlying booster depending on how this method is invoked
        booster = self if isinstance(self, xgb.Booster) else self.model

        # convert to DMatrix
        if isinstance(x, pd.DataFrame):

            # ensure feature alignment
            if hasattr(booster, 'feature_names_in_'):
                x = x.reindex(columns=booster.feature_names_in_, fill_value=0)
            d_test = xgb.DMatrix(x.values.astype(np.float32), feature_names=list(x.columns))
        else:
            d_test = xgb.DMatrix(x.astype(np.float32))

        # get predictions (predicted log survival times)
        pred_survival = booster.predict(d_test)

        # create callable survival functions for each sample
        survival_functions = []
        for i in range(len(pred_survival)):
            # XGBoost AFT model predicts log(time) with normal noise, so we need to convert this to
            # survival probabilities at specific times

            # create a callable survival function
            def survival_function(times, pred=pred_survival[i], scale=self.aft_loss_distribution_alpha):

                # for each time point, calculate survival probability
                # using the cumulative distribution function of the normal distribution

                # convert times to log scale
                log_times = np.log(np.asarray(times))

                # calculate survival probability: 1 - CDF(log(t))
                # where CDF is the cumulative distribution function of the normal distribution
                return 1 - norm.cdf((log_times - pred) / scale)

            # add the survival function to the list
            survival_functions.append(survival_function)
        return survival_functions

    def score(self, x, y):
        booster = self if isinstance(self, xgb.Booster) else self.model

        # get predicted risk scores (negative of predicted log survival time)
        if isinstance(x, pd.DataFrame):

            # ensure feature alignment
            if hasattr(booster, 'feature_names_in_'):
                x = x.reindex(columns=booster.feature_names_in_, fill_value=0)
            d_test = xgb.DMatrix(x.values.astype(np.float32), feature_names=list(x.columns))
        else:
            d_test = xgb.DMatrix(x.astype(np.float32))

        # predict log survival times
        predicted_log_times = booster.predict(d_test)

        # convert to risk scores (negative of predicted log survival time),
        # so that higher risk scores means shorter survival time
        risk_scores = -predicted_log_times

        # extract event and time information
        event = np.array([y_i[0] for y_i in y])
        time = np.array([y_i[1] for y_i in y])

        # calculate concordance index
        c_index = concordance_index_censored(event, time, risk_scores)[0]
        return c_index

    def predict(self, x):
        booster = self if isinstance(self, xgb.Booster) else self.model

        # convert to DMatrix
        if isinstance(x, pd.DataFrame):

            # if the booster has feature names, ensure we only pass those
            if hasattr(self, 'feature_names_in_'):
                x = x.reindex(columns=self.feature_names_in_, fill_value=0)
            elif booster.feature_names is not None:
                x = x.reindex(columns=booster.feature_names, fill_value=0)
            d_test = _xgb.DMatrix(x.values.astype(np.float32), feature_names=list(x.columns))
        else:
            d_test = _xgb.DMatrix(x.astype(np.float32))

        # predict log survival times and convert to risk (negative log-time)
        predicted_log_times = booster.predict(d_test)
        risk_scores = -predicted_log_times
        return risk_scores

    def importance(self, fold=1):
        """
        Calculate feature importance using XGBoost's built-in feature importance.

        This method calculates and displays the importance of each feature in the model.
        Maps XGBoost feature indices (f0, f1, etc.) back to original feature names.
        """

        # get feature importance from XGBoost model
        importance_scores = self.models[fold - 1].get_score(importance_type='gain')

        # map XGBoost feature indices (f0, f1, etc.) to original feature names
        feature_map = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_map[f'f{i}'] = feature_name

        # create a DataFrame for the importance scores with mapped feature names
        importance_df = pd.DataFrame({
            'Feature': [feature_map.get(f, f) for f in importance_scores.keys()],
            'Importance': list(importance_scores.values())
        }).sort_values(by='Importance', ascending=False)

        # display the importance table
        print("\nFeature Importance:")
        print(tabulate(importance_df, headers='keys', tablefmt='fancy_grid'))
        return importance_df

if __name__ == '__main__':
    aft = AcceleratedFailureTime()
    # aft.hpo()
    aft.generate_cv()
    result = aft.evaluate_cv()
    print(f'{aft.name} results:')
    print(f'    Train IBS: {result["train integrated Brier score"]:.4f}')
    print(f'    Test IBS: {result["test integrated Brier score"]:.4f}')

    # aft = AcceleratedFailureTime(partition_frame='ICARE', partition='_icare_disjunction', partition_value=1)
    # aft.generate_cv()
    # aft.evaluate_cv()
    # aft = AcceleratedFailureTime(partition_frame='ICARE', partition='_icare_disjunction', partition_value=0)
    # aft.generate_cv()
    # aft.evaluate_cv()
