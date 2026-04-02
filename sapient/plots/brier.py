import pandas as pd
import plotly.express as px
from sapient.plots.plot import Plot

class Brier(Plot):

    def __init__(self):
        super().__init__()
        self.df = None

    def load_csv(self, name):
        self.df = pd.read_csv(self.data_path / f'{name}.csv')

    def plot(self, name, x='x', y='y', x_domain=None, y_range=None, x_label='Time', y_label='Prediction error',
             color=0, width=9, size=22, splitter=None):
        if splitter is None:

            # use the original implementation for backward compatibility
            fig = px.line()
            trace = {
                'x': self.df[x],
                'y': self.df[y],
                'line': {
                    'color': self.colors[color],
                    'width': width,
                },
            }
            fig.add_traces([trace])
            fig.update_layout(
                xaxis=dict(title=dict(text=f'<b>{x_label}</b>')),
                yaxis=dict(title=dict(text=f'<b>{y_label}</b>')),
                width=800,
                height=600,
            )
            if x_domain is not None:
                fig.update_xaxes(range=[*x_domain])
            if y_range is not None:
                fig.update_yaxes(range=[*y_range])
            return self.finish(fig, name, size=size)
        else:

            # store parameters for use in callback and _plot override
            self.x = x
            self.y = y
            self.x_label = x_label
            self.y_label = y_label
            self.x_domain = x_domain
            self.y_range = y_range

            # set up axis properties for the parent class's plot method
            self.y_axis = y_label  # Set the y-axis label

            # extract x_max from x_domain if provided
            x_max = None
            if x_domain is not None:
                x_max = x_domain[1]

            # use the parent class's plot method which supports splitter
            return super().plot(figure=name, splitter=splitter, size=size, width=width, x_max=x_max, y_range=y_range)

    def callback(self, splitter, index, split):
        """
        Provide data for each splitter value
        Returns x and y data for the specific splitter value
        """
        # Filter the dataframe for the current splitter value
        filtered_df = self.df[self.df[splitter] == split]

        # return the x and y data
        return filtered_df[self.x], filtered_df[self.y]

if __name__ == '__main__':
    brier = Brier()

    # Example 1: Original usage without splitter
    # for bc in [
    #     (1, 1300, 0.3),
    #     (2, 60, 0.25),
    #     (3, 5, 0.2),
    #     (4, 5, 0.14),
    #     (5, 36, 0.18),
    #     (6, 120, 0.14),
    #     (7, 10, 0.28)
    # ]:
    #     print(f'Brier data {bc[0]} ...')
    #     brier.load_csv(f'Brier data {bc[0]}')
    #     brier.plot(f'brier {bc[0]}', x_domain=[0, bc[1]], y_range=[0.0, bc[2]], size=30)

    # Example 2: Using splitter parameter
    # Assuming the CSV has a 'group' column that can be used as a splitter
    print('Brier data with splitter ...')
    brier.load_csv('test_brier_scores')
    # If your CSV doesn't have a splitter column, you might need to add one first
    # For example: brier.df['group'] = brier.df['some_column'].apply(lambda x: 'Group A' if x < 0.5 else 'Group B')
    # brier.df['group'] = brier.df['some_column'].apply(lambda x: 'Group A' if x < 0.5 else 'Group B')
    brier.plot('brier_with_splitter',
               x='x',
               y='y',
               # x_domain=[0, 5],
               y_range=[0.0, 0.14],
               splitter='group',
               size=30
               )
