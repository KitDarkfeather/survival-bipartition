import pandas as pd
from sksurv.nonparametric import kaplan_meier_estimator
from sapient.plots.plot import Plot

class KaplanMeier(Plot):
    y_axis = 'Estimated probability of survival'
    y_format = '.0%'

    def __init__(self, df=None, partition=None, partition_value=1, title=None):
        super().__init__(df, partition, partition_value, title)

    def callback(self, splitter, index, split):
        t, survival_prob = kaplan_meier_estimator(self.df.loc[self.df[splitter] == split, 'event'].astype('bool'),
                                                  self.df.loc[self.df[splitter] == split, 'time'])
        t = pd.Series(t)
        return t, survival_prob

if __name__ == '__main__':
    KaplanMeier('veteran').plot('kaplan-meier', 'days', size=22, width=5)
    KaplanMeier('ELSA').plot('elsa k-m', size=22, width=5)
