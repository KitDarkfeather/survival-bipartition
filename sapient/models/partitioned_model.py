import pandas as pd
from sapient.plots.brier import Brier
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from tabulate import tabulate

class PartitionedModel(object):

    # results
    results = []

    # partition indicators
    partition_indicators = {
        'ICARE': '_icare_disjunction',
        'OC1-u': '_H_star',
        'OC1-c': '_H_star',
        'S-PCA': '_result',
    }

    def __init__(self, model, hyperparameters, partition, ibs=False, hpo=True, verbose=True):
        result = {'partition': partition}

        # unpartitioned model
        df_save = None
        if not ibs:
            m = model(verbose=verbose)
        else:

            # the ibs flag is used by the IBS difference test, which uses bootstrap samples, where there is a
            # need to include partition in the unpartitioned model so that the boostrap dataframe contains it
            m = model(bootstrap=1, partition_frame=partition, partition=self.partition_indicators[partition],
                      verbose=verbose)
            df_save = m.df_bootstrap.copy()
        result['model'] = m.name
        result['model abbreviation'] = m.nickname
        structured_events = m.structured_events
        # calculated_cutoff = m.calculated_cutoff
        for key in hyperparameters:
            setattr(m, key, hyperparameters[key])
        result['hyperparameters'] = ', '.join([f'{key}={hyperparameters[key]}' for key in hyperparameters])
        result['counts: all'] = m.instances
        if hpo:
            m.hpo()
        m.generate_cv()
        full_evaluation = m.evaluate_cv()

        # imminent partition model
        if not ibs:
            m = model(partition_frame=partition, partition=self.partition_indicators[partition], partition_value=1,
                      verbose=verbose)
        else:
            m = model(partition_frame=partition, partition=self.partition_indicators[partition], partition_value=1,
                      bootstrap=2, bootstrap_df=df_save, verbose=verbose)
        for key in hyperparameters:
            setattr(m, key, hyperparameters[key])
        result['counts: imminent'] = m.instances
        if hpo:
            m.hpo()
        m.generate_cv()
        imminent_evaluation = m.evaluate_cv()

        # eventual partition model
        if not ibs:
            m = model(partition_frame=partition, partition=self.partition_indicators[partition], partition_value=0,
                      verbose=verbose)
        else:
            m = model(partition_frame=partition, partition=self.partition_indicators[partition], partition_value=0,
                      bootstrap=2, bootstrap_df=df_save, verbose=verbose)
        for key in hyperparameters:
            setattr(m, key, hyperparameters[key])
        result['counts: eventual'] = m.instances
        if hpo:
            m.hpo()
        m.generate_cv()
        eventual_evaluation = m.evaluate_cv()
        result['counts: combined'] = result['counts: imminent'] + result['counts: eventual']

        # populate the remainder of the result record
        result['concordance: train: all'] = full_evaluation['train c index']
        result['concordance: train: imminent'] = imminent_evaluation['train c index']
        result['concordance: train: eventual'] = eventual_evaluation['train c index']
        result['concordance: train: combined'] = ((result['counts: imminent'] * imminent_evaluation['train c index']
                                                   + result['counts: eventual'] * eventual_evaluation['train c index'])
                                                  / result['counts: all'])
        result['concordance: test: all'] = full_evaluation['test c index']
        result['concordance: test: Uno all'] = concordance_index_ipcw(
            structured_events, structured_events, full_evaluation['test Harrell c index data']['estimate'], 86.5 + 1)[0]
        result['concordance: test: imminent'] = imminent_evaluation['test c index']
        result['concordance: test: eventual'] = eventual_evaluation['test c index']
        result['concordance: test: combined: estimated'] = ((result['counts: imminent']
                                                             * imminent_evaluation['train c index']
                                                             + result['counts: eventual']
                                                             * eventual_evaluation['train c index'])
                                                            / result['counts: all'])
        test_harrell_df = pd.concat([imminent_evaluation['test Harrell c index data'],
                                     eventual_evaluation['test Harrell c index data']])
        result['concordance: test: combined: actual'] = concordance_index_censored(test_harrell_df['event_indicator'],
                                                                                   test_harrell_df['event_time'],
                                                                                   test_harrell_df['estimate'])[0]
        result['ibs: train: all'] = full_evaluation['train integrated Brier score']
        result['ibs: train: imminent'] = imminent_evaluation['train integrated Brier score']
        result['ibs: train: eventual'] = eventual_evaluation['train integrated Brier score']
        result['ibs: train: combined'] = ((result['counts: imminent']
                                           * imminent_evaluation['train integrated Brier score']
                                           + result['counts: eventual']
                                           * eventual_evaluation['train integrated Brier score'])
                                          / result['counts: all'])

        result['ibs: test: all'] = full_evaluation['test integrated Brier score']
        result['ibs: test: imminent'] = imminent_evaluation['test integrated Brier score']
        result['ibs: test: eventual'] = eventual_evaluation['test integrated Brier score']
        result['ibs: test: combined'] = ((result['counts: imminent']
                                          * imminent_evaluation['test integrated Brier score']
                                          + result['counts: eventual']
                                          * eventual_evaluation['test integrated Brier score'])
                                         / result['counts: all'])
        result['concordance: combined v. all'] = (result['concordance: test: combined: actual']
                                                  / result['concordance: test: all'] - 1) * 100.0
        result['ibs: combined v. all'] = -(result['ibs: test: combined'] / result['ibs: test: all'] - 1) * 100.0
        self.results.append(result)

        # generate a plot comparing partitioned and unpartitioned model Brief scores over time

        # add a group to unpartitioned model Brier scores
        df_unpartitioned = full_evaluation['Brier score over time'].copy()
        df_unpartitioned['group'] = 'not partitioned'

        # combined imminent and eventual Brier scores and add a group
        df_imminent = imminent_evaluation['Brier score over time']
        df_eventual = eventual_evaluation['Brier score over time']
        df_partitioned = df_imminent[['x']]
        df_partitioned['y'] = ((result['counts: imminent'] * df_imminent['y']
                                + result['counts: eventual'] * df_eventual['y'])
                               / result['counts: all'])
        df_partitioned['group'] = 'partitioned'

        # generate Brief score plot for model and partition
        brier = Brier()
        brier.df = pd.concat([df_partitioned, df_unpartitioned])
        brier.plot(f'{result['model abbreviation']}_{partition}_brier_scores',
                   x_label='Time in months', y_label='Brier score', splitter='group',
                   x_domain=[0, 86.5], y_range=[0.0, brier.df['y'].max() * 1.1], width=6, size=22)

    @classmethod
    def show_results(cls):

        # write results to dataframe

        # create the columns
        results_df = pd.DataFrame({
            '\n\nPartition': [result['partition'] for result in cls.results],
            '\n\nModel': [result['model'] for result in cls.results],
            '\n\nHyperparameters': [result['hyperparameters'] for result in cls.results],
            '\nTotal\ncount': [f'{result['counts: all']:,}' for result in cls.results],
            '\nImminent\ncount': [f'{result['counts: imminent']:,}' for result in cls.results],
            '\nEventual\ncount': [f'{result['counts: eventual']:,}' for result in cls.results],
            'Total\ntest\nconcordance': [f'{result['concordance: test: all']:.3f}' for result in cls.results],
            'Imminent\ntest\nconcordance': [f'{result['concordance: test: imminent']:.3f}' for result in cls.results],
            'Eventual\ntest\nconcordance': [f'{result['concordance: test: eventual']:.3f}' for result in cls.results],
            'Combined test\nconcordance\n(actual)': [f'{result['concordance: test: combined: actual']:.3f}'
                                                     for result in cls.results],
            'Combined v.\ntotal test\nconcordance': [f'{result['concordance: combined v. all']:.1f}%'
                                                for result in cls.results],
            'Total\ntest\nIBS': [f'{result['ibs: test: all']:.3f}' for result in cls.results],
            'Imminent\ntest\nIBS': [f'{result['ibs: test: imminent']:.3f}' for result in cls.results],
            'Eventual\ntest\nIBS': [f'{result['ibs: test: eventual']:.3f}' for result in cls.results],
            'Combined\ntest\nIBS': [f'{result['ibs: test: combined']:.3f}' for result in cls.results],
            'Combined v.\ntotal test\nIBS': [f'{result['ibs: combined v. all']:.1f}%'
                                       for result in cls.results],
        })

        # tabulate results
        print("\nPARTITIONED MODEL RESULTS")
        print(tabulate(results_df, headers='keys', tablefmt='fancy_grid', showindex=False))
        print(tabulate(results_df, headers='keys', tablefmt='latex_longtable', showindex=False))


if __name__ == '__main__':
    from sapient.models.cox_elastic_net import CoxElasticNet
    from sapient.models.random_survival_forest import RandomSurvivalForest
    from sapient.models.accelerated_failure_time import AcceleratedFailureTime

    # generate full results for 4 partitioning methods and 3 modelling methods
    for _partition in [
        'ICARE',
        # 'OC1-u',
        'OC1-c',
        'S-PCA'
    ]:
        for _model in [
            (CoxElasticNet, {}),
            (AcceleratedFailureTime, {}),
            (RandomSurvivalForest, {}),
        ]:
            PartitionedModel(_model[0], _model[1], _partition, hpo=False)
    PartitionedModel.show_results()
