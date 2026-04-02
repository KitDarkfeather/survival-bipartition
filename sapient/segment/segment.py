from umberto.datasets.elsa import ELSA
from datetime import datetime
from itertools import combinations
import json
import numpy as np
from pathlib import Path
import prince
from random import choice, sample, seed, uniform
from sapient.analytics import Analytics
from sapient.plots.kaplan_meier import KaplanMeier
from scipy.optimize import minimize
from scipy.stats import t as t_distribution
from sksurv.nonparametric import kaplan_meier_estimator
import warnings

class Segment(Analytics):
    """
    Partition the imminent and the eventual by maximising the area between the survival curves.
    """

    # hyperparameters
    last_death = 86.5
    min_partition = 40
    max_values = 100_000
    prior_lambda = 0.00
    min_instances = 0
    max_median = 100

    def __init__(self, training='sa_training_subset', training_value=1, partition_frame=None, partition=None):
        super().__init__(training, training_value, partition_frame, partition)
        self.ordered_splits = None
        self.last_lambda = None
        self.numeric_features_plugged = [feature for feature in self.numeric_features_plugged
                                         if feature not in ['sa_age_squared']]

    def _moment_arrays(self, time, event, t_max):
        if t_max is None:
            t_max = self.last_death
        t, survival_prob = kaplan_meier_estimator(event.astype('bool'), time)
        t = np.concatenate([np.array([0.0]), t, np.array([t_max])])
        survival_prob = np.concatenate([np.array([1.0]), survival_prob, np.array([survival_prob[-1]])])
        try:
            n = np.where(t == t_max)[0][0]
        except IndexError:
            n = len(t) - 1
        return n, t, survival_prob

    def _km_area(self, time, event, t_max=None):
        n, t, survival_prob = self._moment_arrays(time, event, t_max)
        area = 0
        for i in range(n):
            area += survival_prob[i] * (t[i + 1] - t[i])
        return area

    def _km_second_moment(self, time, event, t_max=None):
        n, t, survival_prob = self._moment_arrays(time, event, t_max)
        second_moment = 0
        for i in range(n):
            second_moment += t[i] * survival_prob[i] * (t[i + 1] - t[i])
        return 2 * second_moment

    def _t_calculations(self, partition):

        # E(T) = integral(0, inf, S(t))
        # E(T^2) = 2 * integral(0, inf, t * S(t))
        # the mean and variance of T are given in eq. 2.4.2 and 2.4.3 (p. 33) of Klein and Moeschberger
        # Mu(T) = E(T)
        # Var(T) = E(T^2) - E(T)^2

        # n
        n = self.df[self.df['partition'] == partition].shape[0]
        # DEBUG: print(f'n = {n} for partition 0')

        # mu hat
        mu = self._km_area(self.df.loc[self.df['partition'] == partition, 'time'],
                           self.df.loc[self.df['partition'] == partition, 'event'])
        # DEBUG: print(f'E(T) = {mu} for partition 0')

        # variance with sample correction
        sm = self._km_second_moment(self.df.loc[self.df['partition'] == partition, 'time'],
                                    self.df.loc[self.df['partition'] == partition, 'event'])
        # DEBUG: print(f'E(T2) = {sm} for partition 0')
        var = sm - mu ** 2
        var *= n / (n - 1)
        # DEBUG: print(f'var(T) = {var} for partition 0')
        return n, mu, var

    def t_test(self):
        """
        Simple two-sample t-test for unequal variances.  We start with a naive two-sample t-test, and will use it as a
        benchmark to compare to more elaborate methods.

        A better test could be based on the sample mean and variance given in eq. 4.5.1 and 4.5.2, pp. 118-9 of Klein
        and Moeschberger.

        The standard error of the mean is given in eq. 4.5.3, pp. 118-9 of Klein and Moeschberger.

        The standard error of the restricted error under the survival curve is given in Collett p. 340.

        A still better test of mu1 - mu2 is given in Klein and Moeschberger pp 229-30; the K-M curves are not
        restricted, but weights based on p(censorship) are used (which for us means weights after 86.5 would be 0);
        we would want to code this test and verify that weights after 86.5 are 0.

        """
        p = None
        try:
            n_0, mu_0, var_0 = self._t_calculations(0)
            n_1, mu_1, var_1 = self._t_calculations(1)

            # calculate the t-statistic
            t = (mu_1 - mu_0) / (np.sqrt(var_1 / np.sqrt(n_1) + var_0 / np.sqrt(n_0)))
            # print(f't = {t}')

            # calculate the degrees of freedom
            dof = ((var_1 / n_1 + var_0 / n_0) ** 2
                   / ((var_1 ** 2) / ((n_1 ** 2) * (n_1 - 1)) + (var_0 ** 2) / ((n_0 ** 2) * (n_0 - 1))))
            # print(f'dof = {dof}')

            # calculate the p-value
            p = 2 * t_distribution.sf(abs(t), dof)
            # DEBUG: print(f'p = {p}')
        except KeyError:
            print('No partition specified.')
        return p

    def objective(self, partition, pure=False):
        pq = self.df[partition].value_counts(normalize=True)
        try:
            a0 = self._km_area(self.df.loc[self.df[partition] == 0, 'time'],
                               self.df.loc[self.df[partition] == 0, 'event'])
            a1 = self._km_area(self.df.loc[self.df[partition] == 1, 'time'],
                               self.df.loc[self.df[partition] == 1, 'event'])
            area = float(a0 - a1)
            if pure is True:
                result = area
            else:
                prior_constraint = (pq.iloc[0] * pq.iloc[1]) ** self.prior_lambda
                result = area * prior_constraint
        except ValueError:
            result = 0.0
        return result

    @staticmethod
    def _partitions(values):
        partitions = set([])
        for left_size in range(1, len(values)):

            # get all subsets of size left_size
            lefts = list(combinations(values, left_size))

            # generate and add the partitions
            for left in lefts:
                left = set(left)
                right = set(values) - left
                left = tuple(sorted(left))
                right = tuple(sorted(right))
                partitions.add(tuple(sorted([left, right])))
        result = []
        for p in partitions:
            partition = [list(item) for item in p]
            result.append(partition)
        return result

    def find_best_split(self, partition):
        print(f'Finding area for "{self.features[partition]['description']}" ...')
        result = None
        imminent = None
        best_p = None

        # process each feature type appropriately
        if self.features[partition]['level'] == 'boolean':
            best_split = 0
            result = 0

            # make sure partition is large enough
            if min(self.df[partition].value_counts()) >= self.min_partition:
                result = self.objective(partition)
            if result > 0:
                imminent = ' == 1'
                self.df[f'best_{partition}'] = self.df[partition]
            else:
                imminent = ' == 0'
                self.df[f'best_{partition}'] = self.df[partition].map({0: 1, 1: 0})
            # DEBUG: print(f'best_{partition}', self.df[f'best_{partition}'].value_counts())
        elif self.features[partition]['level'] in ['categorical', 'ordinal']:

            # get all k available values
            values =self.df[partition].unique()

            # generate all possible partitions of size 2
            partitions = self._partitions(values)

            # initialise best split as empty set
            best_split = []
            best_area = 0
            best_p = -1

            # loop through the subsets
            for p in partitions:
                self.df[f'best_{partition}'] = self.df[partition].map(lambda x: 1 if x in p[0] else 0)
                if (len(self.df[f'best_{partition}'].value_counts()) > 1) and (
                        min(self.df[f'best_{partition}'].value_counts()) >= self.min_partition):
                    result = self.objective(f'best_{partition}')

                    # save new best
                    if abs(result) > abs(best_area):
                        best_area = result
                        best_split = p
                        best_p = self.df[f'best_{partition}'].value_counts(normalize=True)[0]

            # save best
            result = best_area
            p = best_split
            if best_area != 0:
                self.df[f'best_{partition}'] = self.df[partition].map(lambda x: 1 if x in best_split[0] else 0)
            try:
                if best_area > 0:
                    imminent = f' in {best_split[0]}'
                else:
                    imminent = f' in {best_split[1]}'
                    self.df[f'best_{partition}'] = self.df[f'best_{partition}'].map({0: 1, 1: 0})
            except IndexError:
                pass
        elif self.features[partition]['level'] == 'ratio':

            # get all available values and sort them
            values = sorted(self.df[partition].unique())

            # subset the list to speed up the search
            if len(values) > self.max_values:
                values = sorted(sample(values, self.max_values))

            # initial best split point
            best_split = values[0] - 1
            best_area = 0

            # loop through the values
            for v in values[:-1]:
                self.df[f'best_{partition}'] = self.df[partition].map(lambda x: 1 if x <= v else 0)
                self.df = self.df.copy()
                if (len(self.df[f'best_{partition}'].value_counts()) > 1) and (
                        min(self.df[f'best_{partition}'].value_counts()) >= self.min_partition):
                    result = self.objective(f'best_{partition}')

                    # save new best
                    if abs(result) > abs(best_area):
                        best_area = result
                        best_split = v
            # DEBUG: print(f'Regular best split: {best_split} {best_area} (in {timer() - start})')

            # determine imminent side and set imminent appropriately
            self.df[f'best_{partition}'] = self.df[partition].map(lambda x: 1 if x <= best_split else 0)
            if best_area > 0:
                imminent = f' <= {best_split}'
            else:
                imminent = f' > {best_split}'
                self.df[f'best_{partition}'] = self.df[f'best_{partition}'].map({0: 1, 1: 0})
            result = best_area
        else:
            raise Exception(f'Unknown feature form or level or combination')
        try:
            p = self.df[f'best_{partition}'].value_counts(normalize=True).iloc[1]
        except IndexError:
            p = best_p
        self.df = self.df.copy()
        return imminent, result, p

    def _partition_of(self, b, partition):
        self.df['_temp'] = self.df[partition].map(lambda x: 1 if x <= b[0] else 0)
        if (len(self.df['_temp'].value_counts()) > 1) and (
                min(self.df['_temp'].value_counts()) >= self.min_partition):
            result = -abs(self.objective('_temp'))
        else:
            result = 0
        return result

    def get_ordered_splits(self):
        results = None
        split_text = f'split_{int(self.prior_lambda * 100 + 0.01)}'
        json_path = Path(__file__).resolve().parent / 'json' / f'{split_text}.json'
        parquet_exists = True
        try:
            self.elsa.load(f'ELSA_{split_text}')
        except KeyError:
            parquet_exists = False
        if json_path.is_file() and parquet_exists:
            with open(json_path, 'r') as json_file:
                splits = json.load(json_file)
                results = list(zip(splits['absolute areas'], splits['proportions'], splits['areas'], splits['features'],
                                   splits['relations'], splits['descriptions'],))
            self.df = self.elsa.dfs[f'ELSA_{split_text}']
        else:

            # find the best split for each feature
            results = []
            for feature in self.features:
                if self.features[feature]['category'] not in ['target', 'validation']:
                    i_group, r, p = self.find_best_split(feature)
                    if r is not None and abs(r) > 0:
                        results.append((float(abs(r)), float(p), float(r), feature, i_group,
                                        self.features[feature]['description']))

            # preserve the results
            if self.max_values >= len(self.df):
                splits = {
                    'descriptions': [result[5] for result in results],
                    'features': [result[3] for result in results],
                    'relations': [result[4] for result in results],
                    'proportions': [result[1] for result in results],
                    'absolute areas': [result[0] for result in results],
                    'areas': [result[2] for result in results],
                }
                with open(json_path, 'w') as json_file:
                    # noinspection PyTypeChecker
                    json.dump(splits, json_file)
                self.elsa.save(self.df, f'ELSA_{split_text}')
        results.sort(reverse=True)

        # display the results
        for result in results:
            print(result)

        # return the results
        return results

    def load_splits(self):
        try:
            with open(Path(__file__).resolve().parent / 'json' / 'split.json', 'r') as json_file:
                splits = json.load(json_file)
                splits = list(zip(splits['absolute areas'], splits['proportions'], splits['areas'], splits['features'],
                                  splits['relations'], splits['descriptions'],))
        except FileNotFoundError:
            splits = self.get_ordered_splits()
        return splits

    @staticmethod
    def hyperbox(x, y):

        # calculate the effect of leaving each point out
        areas = []
        for i in range(len(x)):

            # create arrays with one knocked out
            x_ = x[:]
            del x_[i]
            y_ = y[:]
            del y_[i]

            # calculate area after knockout
            area = 0
            x_ = [0] + x_
            y_ = [0] + y_
            for i in range(1, len(x_)):
                area += y_[i] * (x_[i] - x_[i - 1])
            areas.append(area)
        print(areas)
        print(min(areas))
        index = areas.index(min(areas))
        print(index)
        print(x[index], y[index])
        return index

    def icare(self, prior_lambda=0.00):
        seed(42)
        self.prior_lambda = prior_lambda

        # need to regenerate splits so that the partitions for current lambda are added to the dataframe
        if self.ordered_splits is None or self.last_lambda != self.prior_lambda:
            self.ordered_splits = self.get_ordered_splits()
            self.last_lambda = self.prior_lambda

        # initialise the disjunction partition
        self.df['_icare_disjunction'] = 0
        trade_offs = {
            'features': [],
            'instances': [],
            'areas': [],
        }

        # generate the cumulative disjunction partitions
        k = 0
        for partition in [ordered_split[3] for ordered_split in self.ordered_splits]:

            # build the disjunction
            k += 1
            print(f'Next partition {k}: {partition}')
            if self.df['_icare_disjunction'].sum() == 0:
                self.df['_icare_disjunction'] = self.df[f'best_{partition}']
            else:
                self.df['_icare_disjunction'] = self.df[f'best_{partition}'] | self.df[f'_icare_disjunction']
            try:
                area = self.objective('_icare_disjunction')
            except ValueError:
                break
            trade_offs['features'].append(partition)
            trade_offs['instances'].append(self.df['_icare_disjunction'].sum())
            trade_offs['areas'].append(area)
        for y, x in zip(trade_offs['areas'], trade_offs['instances']):
            print(f'{y}|{x}')

        # find the optimal point on the trade-off curve with at least minimum instances
        first_index = next(x[0] for x in enumerate(trade_offs['instances']) if x[1] > self.min_instances)
        index = self.hyperbox(trade_offs['instances'][first_index:], trade_offs['areas'][first_index:]) + first_index
        print(f'The optimal point is {trade_offs['instances'][index]}, {trade_offs['areas'][index]}')

        # generate the K-M curves
        print(trade_offs['features'][:index + 1])

        # recreate the oblique split
        self.df['_icare_disjunction'] = self.df[f'best_{trade_offs['features'][0]}']
        for partition in trade_offs['features'][1:index + 1]:
            self.df['_icare_disjunction'] = self.df[f'best_{partition}'] | self.df[f'_icare_disjunction']

        # save the results
        icare_run = {
            'parameters': {
                'maximum values': self.max_values,
                'prior lambda': self.prior_lambda,
            },
            'results': {
                'area': self.objective('_icare_disjunction', True),
                'instances': int(self.df['_icare_disjunction'].sum()),
            },
            'features': trade_offs['features'][:index + 1],
        }
        with open(Path(__file__).resolve().parent / 'json' / f'icare {datetime.now().strftime(
                '%Y-%m-%d-%H-%M-%S')}.json', 'w') as json_file:
            # noinspection PyTypeChecker
            json.dump(icare_run, json_file)

        # generate the plot
        self.df[f'_icare_disjunction_text'] = self.df[f'_icare_disjunction'].map({1: 'imminent', 0: 'eventual'})
        KaplanMeier(self.df).plot(f'icare {datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}',
                                  splitter='_icare_disjunction_text', unit='months', size=22, width=5)

        # save the results
        print(f'ELSA after ICARE has {len(self.df.columns)} columns.')
        ELSA().save(self.df, 'ELSA-ICARE')

    def oc1(self, prior_lambda=0.32, depth=100_000, knockouts=3, max_feature_count=None, numeric=False):
        seed(42)
        self.prior_lambda = prior_lambda

        # need to regenerate splits so that the partitions for current lambda are added to the dataframe
        if self.ordered_splits is None or self.last_lambda != self.prior_lambda:
            self.ordered_splits = self.get_ordered_splits()[:depth]
            self.last_lambda = self.prior_lambda

            # centre the features
            for feature in [f'best_{ordered_split[3]}' for ordered_split in self.ordered_splits]:
                self.df[feature] = self.df[feature].map({1: 1, 0: -1})
                self.df = self.df.copy()
            if numeric is True:
                for feature in self.numeric_features_plugged:
                    self.df[feature] -= self.df[feature].median()
                    self.df = self.df.copy()

        # initialise the oblique hyperplane H∗ with the best axis-parallel split
        first_splitter = f'best_{self.ordered_splits[0][3]}'
        weights = {
            'intercept': 0.0,
            first_splitter: 1.0
        }
        self.df['_H_star'] = self.df[first_splitter]
        maximum = self.ordered_splits[0][0]
        instances = self.df[f'best_{self.ordered_splits[0][3]}'].sum()
        print(f'Area: {maximum}, instances: {instances}')
        print(self.df['_H_star'].value_counts())

        # step 1
        print('STEP 1')
        weights, maximum = self._oc1_part_1(weights, maximum, initial_set={first_splitter},
                                            knockouts=knockouts, max_feature_count=max_feature_count, numeric=numeric)
        true_area = self.objective('_H_star', True)
        instances = int(self.df['_H_star'].sum())
        print(f'The maximum value is {true_area}.')
        print(f'    representing {instances} instances')

        # steps 2 and 3
        for sub in range(5):

            # step 2
            print(f'STEP 2{'abcde'[sub]}')
            new_weights, new_maximum = self._oc1_part_2(dict(weights), maximum, numeric=numeric)
            if new_maximum > maximum:
                maximum = new_maximum
                weights = new_weights
                self._oc1_h_star(weights)
                true_area = self.objective('_H_star', True)
                instances = int(self.df['_H_star'].sum())
                print(f'The maximum value is {true_area}.')
                print(f'    representing {instances} instances')

                # step 3
                print(f'STEP 3{'abcde'[sub]}')
                weights, maximum = self._oc1_part_1(weights, maximum, max_feature_count=max_feature_count,
                                                    numeric=numeric)
                self._oc1_h_star(weights)
                true_area = self.objective('_H_star', True)
                instances = int(self.df['_H_star'].sum())
                print(f'The maximum value is {true_area}.')
                print(f'    representing {instances} instances')
            else:
                print('Step 2 failed to improve solution')
        self._oc1_h_star(weights)
        true_area = self.objective('_H_star', True)
        instances = int(self.df['_H_star'].sum())
        print(f'The maximum value is {true_area}.')
        print(f'    representing {instances} instances')
        print(f'intercept: {weights['intercept']}')
        factors = 0
        for key in weights:
            if key != 'intercept':
                if weights[key] > 0:
                    factors += 1
                    print(f'{key}: {weights[key]}')

        # generate the plot
        self.df['_H_star_text'] = self.df['_H_star'].map({1: 'imminent', 0: 'eventual'})
        km = KaplanMeier(self.df)
        html_path = km.plot(f'oc1 {datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}',
                splitter='_H_star_text', unit='months', size=22, width=5)

        # output the results
        median = km.medians[-1] if len(km.medians) > 0 else None
        if median is None:
            median = 100
        print(f'Median: {median}')
        print(f'Instances: {instances}')
        print(median, self.max_median, instances, self.min_instances)
        if median < self.max_median and instances > self.min_instances:
            oc1_run = {
                'results': {
                    'area': true_area,
                    'instances': instances,
                    'factors': factors,
                    'median': km.medians[-1] if len(km.medians) > 0 else None,
                },
                'parameters': {
                    'maximum values': self.max_values,
                    'prior lambda': self.prior_lambda,
                    'depth': depth,
                    'knockouts': knockouts,
                    'max feature count': max_feature_count,
                    'numeric': numeric,
                },
                'weights': dict([(key, weights[key]) for key in weights if weights[key] != 0.0]),
            }
            with open(Path(__file__).resolve().parent / 'json'
                      / f'oc1 {datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json', 'w') as json_file:
                # noinspection PyTypeChecker
                json.dump(oc1_run, json_file)
        else:
            Path(html_path).unlink()

        # save the results
        print(f'ELSA after OCl has {len(self.df.columns)} columns.')
        if max_feature_count is None:
            ELSA().save(self.df, 'ELSA-OC1-u')
        else:
            ELSA().save(self.df, 'ELSA-OC1-c')

    def _oc1_h_star(self, weights, display=False):
        self.df['_H_star'] = weights['intercept']
        if display is True:
            print(f'intercept: {weights["intercept"]}')
        for x_k in weights:
            if x_k != 'intercept':
                if (display is True) and (weights[x_k] > 0):
                    print(f'{x_k}: {weights[x_k]}')
                self.df['_H_star'] += weights[x_k] * self.df[x_k]
        self.df['_H_star'] = self.df['_H_star'].map(lambda x: 1 if x > 0 else 0)

    def _oc1_of1(self, w, x_k, weights):
        self.df['_oc1'] = w[0] + w[1] * self.df[x_k]
        for x_i in weights:
            if x_i != 'intercept':
                self.df['_oc1'] += weights[x_i] * self.df[x_i]
        self.df['_oc1'] = self.df['_oc1'].map(lambda x: 1 if x > 0 else 0)
        return -self.objective('_oc1')

    @staticmethod
    def _oc1_current_feature_count(weights):
        return sum([1 for key in weights if key != 'intercept' and weights[key] != 0.0])

    def _oc1_part_1(self, weights, maximum, initial_set=None, knockouts=0, max_feature_count=None, numeric=False):
        if initial_set is None:
            initial_set = set()
        attempted = initial_set
        iteration = 0
        w_0 = weights['intercept']
        remaining_knockouts = knockouts
        while True:
            improved = False
            iteration += 1
            print(f'Iteration {iteration} ...')
            if numeric is True:
                numeric_features = self.numeric_features_plugged
            else:
                numeric_features = []
            if max_feature_count is None:
                max_feature_count = len([f'best_{ordered_split[3]}' for ordered_split in self.ordered_splits]
                                        + numeric_features)
            potential = set([f'best_{ordered_split[3]}' for ordered_split in self.ordered_splits]
                            + numeric_features) - attempted
            total = len(potential)
            sub_iteration = 0
            while len(potential) > 0 and self._oc1_current_feature_count(weights) < max_feature_count:
                sub_iteration += 1
                if sub_iteration % (total // 5) == 0:
                    print(f'Step {sub_iteration} of {total} ...')

                # make random choice deterministic with seed and sorted
                x_k = choice(list(sorted(potential)))

                # update H with x_k
                w_k = 1.0
                result = minimize(self._oc1_of1,
                                  np.array([w_0, w_k]),
                                  args=(x_k, weights),
                                  method='BFGS',
                                  options={}
                                  )
                w_0_old = w_0
                w_0 = result.x[0]
                w_k = result.x[1]
                weights['intercept'] = w_0
                if x_k in weights:
                    weights[x_k] += w_k
                else:
                    weights[x_k] = w_k
                self._oc1_h_star(weights)
                area_k = self.objective('_H_star')
                if area_k > maximum:

                    # keep the improvement
                    print(f'SOLUTION IMPROVED: Area: {area_k}, instances: {self.df['_H_star'].sum()}, feature: {x_k}')
                    maximum = area_k
                    improved = True
                    print(f'{self._oc1_current_feature_count(weights)} features at iteration {sub_iteration} ...')
                else:

                    # back out
                    weights['intercept'] = w_0_old
                    weights[x_k] -= w_k
                    if weights[x_k] == np.float64(0.0):
                        del weights[x_k]
                    self._oc1_h_star(weights)
                potential -= {x_k}
            if improved is False:

                # attempt to perturb solution by knocking out a weight
                if remaining_knockouts > 0:
                    candidate = choice([feature for feature in weights
                                        if feature != 'intercept' and weights[feature] != 0])
                    print(f'Knocking out: {candidate} ...')
                    del weights[candidate]
                    self._oc1_h_star(weights)
                    maximum = self.objective('_H_star')
                    print(f'ADJUSTED SOLUTION: Area: {maximum}, instances: {self.df['_H_star'].sum()}')
                    remaining_knockouts -= 1
                else:
                    break
            attempted = set()
        return weights, maximum

    def _oc1_of2(self, alpha, weights, perturb):
        test_weights = {}
        for key in weights:
            test_weights[key] = weights[key] + alpha[0] * perturb[key]
        self.df['_oc1'] = test_weights['intercept']
        for x_i in test_weights:
            if x_i != 'intercept':
                self.df['_oc1'] += test_weights[x_i] * self.df[x_i]
        self.df['_oc1'] = self.df['_oc1'].map(lambda x: 1 if x > 0 else 0)
        return -self.objective('_oc1')# + alpha[0] ** 2

    def _oc1_part_2(self, weights, maximum, numeric=False):
        benchmark = maximum

        # attempt to find a random hyperplane that beats the benchmark
        for iteration in range(1, 201):
            if iteration % 40 == 0:
                print(f'Iteration {iteration} ...')

            # generate a random hyperplane
            perturb = {}
            for key in weights:
                perturb[key] = weights[key] * uniform(0, 1) + uniform(0, 1)

            # initialise alpha
            alpha = 1.0

            # find the alpha that maximises area for H_star + alpha * H_random
            result = minimize(self._oc1_of2, np.array([alpha]),
                              args=(weights, perturb),
                              # method='Powell',
                              # method='Nelder-Mead',
                              # options={'xatol': 1e-8,},
                              method = 'BFGS',
                              )
            alpha = result.x[0]
            for key in weights:
                weights[key] += perturb[key] * alpha

            # prevent weight blowup
            if numeric is False:
                adjustment = min([abs(weights[key]) for key in weights])
                for key in weights:
                    weights[key] /= adjustment
            self._oc1_h_star(weights)
            maximum = self.objective('_H_star')
            if maximum > benchmark:
                print(f'Alpha: {alpha}')
                break
        return weights, maximum

    def spca(self, prior_lambda=None, cutoff=None, q_maximum=94, components=5):
        seed(42)
        if not isinstance(prior_lambda, list):
            if prior_lambda is None:
                prior_lambda = [0.00]
            else:
                prior_lambda = [prior_lambda]
        if not isinstance(cutoff, list):
            if cutoff is None:
                cutoff = [0.10]
            else:
                cutoff = [cutoff]

        # cycle through lamba values
        for pl in prior_lambda:
            self.prior_lambda = pl

            # need to regenerate splits so that the partitions for current lambda are added to the dataframe
            if self.ordered_splits is None or self.last_lambda != self.prior_lambda:
                self.ordered_splits = self.get_ordered_splits()
                self.last_lambda = self.prior_lambda

            # interleave numeric and nominal features into features
            all_features = []
            for feature in [ordered_split[3] for ordered_split in self.ordered_splits]:
                all_features.append(f'best_{feature}')
                if feature in self.numeric_features_plugged:
                    all_features.append(feature)
                if feature in self.nominal_features:
                    all_features.append(feature)

            # cycle through cutoff values
            for co in cutoff:

                # cut off features
                features = all_features[:int(len(all_features) * co)]
                print(features)
                if components == -1:
                    components = len(features)

                # transform Boolean features so Prince can recognise them as such
                for feature in features:
                    if feature.startswith('best_'):
                        self.df[feature] = self.df[feature].map({0: False, 1: True})
                        self.df = self.df.copy()

                # perform the fa_md
                warnings.filterwarnings('ignore')
                fa_md = prince.FAMD(
                    n_components=components,
                    n_iter=1000,
                    copy=True,
                    check_input=True,
                    random_state=42,
                    engine='sklearn',
                    handle_unknown='error',
                )
                warnings.filterwarnings('default')
                try:
                    fa_md = fa_md.fit(self.df[features])
                    # print(f'DEBUG: {fa_md.eigenvalues_summary}')

                    # find the greatest area in the acceptable range
                    for component in range(components):
                        self.df[f'component {component}'] = fa_md.row_coordinates(self.df)[component]
                        self.df = self.df.copy()
                    max_area = 0.0

                    # one component at a time
                    best_slope = 0
                    best_threshold = None
                    best_q = None
                    best_component = None
                    for component in range(components):
                        for q in [x / 100 for x in range(5, q_maximum + 1)]:
                            threshold = self.df[f'component {component}'].quantile(q)
                            self.df['_test'] = self.df[f'component {component}'].map(lambda x: 1 if x > threshold else 0)
                            self.df = self.df.copy()
                            area = self.objective('_test')
                            if area > max_area:
                                max_area = area
                                best_threshold = threshold
                                best_q = q
                                best_component = component
                    self.df['_result'] = self.df[f'component {best_component}'].map(
                        lambda x: 1 if x > best_threshold else 0)
                    self.df = self.df.copy()
                    print(f'Best threshold: {best_threshold}')

                    # generate the curve
                    self.df['_result_text'] = self.df['_result'].map({1: 'imminent', 0: 'eventual'})
                    self.df = self.df.copy()
                    km = KaplanMeier(self.df)

                    # display the curve
                    html_path = km.plot(f'famd {datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}',
                            splitter='_result_text', unit='months', size=22, width=5)

                    # test the results
                    median = km.medians[-1] if len(km.medians) > 0 else None
                    if median is None:
                        median = 100
                    instances = int(self.df['_result'].sum())
                    print(f'Median: {median}')
                    print(f'Instances: {instances}')
                    print(median, self.max_median)
                    print(instances, self.min_instances)
                    if median < self.max_median and instances > self.min_instances:

                        # save the results
                        fa_md_run = {
                            'results': {
                                'area': self.objective('_result', True),
                                'instances': instances,
                                'features': len(features),
                                'component': best_component + 1,
                                'median': median,
                            },
                            'parameters': {
                                'prior lambda': self.prior_lambda,
                                'cutoff': co,
                                'q': best_q,
                                'm': best_slope,
                                'b': best_threshold,
                                'components': components,
                            },
                        }
                        with open(
                                Path(__file__).resolve().parent / 'json'
                                / f'fa_md {datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json', 'w') as json_file:
                            # noinspection PyTypeChecker
                            json.dump(fa_md_run, json_file)

                        # save the results
                        ELSA().save(self.df, 'ELSA-S-PCA')
                    else:
                        Path(html_path).unlink()
                except ValueError:
                    print('FAMD not run. All variables are qualitative.')

if __name__ == '__main__':
    segment = Segment()

    # ICARE with defaults
    segment.min_instances = 200
    segment.icare()

    # OC1
    segment.oc1(prior_lambda=0.3, numeric=False, knockouts=3)

    # OC1-maxed
    segment.oc1(prior_lambda=.32, max_feature_count=7, knockouts=1, numeric=False)

    # SPCA
    segment.max_median = 75
    segment.min_instances = 350
    segment.spca(prior_lambda=[0.28], cutoff=[0.17])

    # calculate partition p-values
    print('\nICARE')
    icare_segment = Segment(partition_frame='ICARE', partition='_icare_disjunction')
    print(f'p = {icare_segment.t_test()}')
    print('\nOC1')
    oc1_segment = Segment(partition_frame='OC1-c', partition='_H_star')
    print(f'p = {oc1_segment.t_test()}')
    print('\nSPCA')
    spca_segment = Segment(partition_frame='S-PCA', partition='_result')
    print(f'p = {spca_segment.t_test()}')
