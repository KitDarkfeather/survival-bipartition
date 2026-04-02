from umberto.datasets.datatables import Datatables
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly
from scipy.signal import savgol_filter

class Plot(object):
    base_path = Path(__file__).resolve().parent
    data_path = base_path / 'data'
    image_path = base_path / 'images'
    background = 'white'
    colors = ['#007FBF', 'orange', 'navy', 'blueviolet', 'cadetblue', 'cornflowerblue', 'gray']
    bar_colors = ['navy', 'red']
    # colors = ['blue', 'navy', 'blueviolet', 'cadetblue', 'cornflowerblue',]
    annotations = 'lightslategray'
    annotation_lines = 'lightgray'
    gridlines = 'lightgray'
    # aliceblue, antiquewhite, aqua, aquamarine, azure, beige, bisque, black, blanchedalmond, blue,
    # blueviolet, brown, burlywood, cadetblue, chartreuse, chocolate, coral, cornflowerblue,
    # cornsilk, crimson, cyan, darkblue, darkcyan, darkgoldenrod, darkgray, darkgrey, darkgreen,
    # darkkhaki, darkmagenta, darkolivegreen, darkorange, darkorchid, darkred, darksalmon, darkseagreen,
    # darkslateblue, darkslategray, darkslategrey, darkturquoise, darkviolet, deeppink, deepskyblue,
    # dimgray, dimgrey, dodgerblue, firebrick, floralwhite, forestgreen, fuchsia, gainsboro,
    # ghostwhite, gold, goldenrod, gray, grey, green, greenyellow, honeydew, hotpink, indianred, indigo,
    # ivory, khaki, lavender, lavenderblush, lawngreen, lemonchiffon, lightblue, lightcoral, lightcyan,
    # lightgoldenrodyellow, lightgray, lightgrey, lightgreen, lightpink, lightsalmon, lightseagreen,
    # lightskyblue, lightslategray, lightslategrey, lightsteelblue, lightyellow, lime, limegreen,
    # linen, magenta, maroon, mediumaquamarine, mediumblue, mediumorchid, mediumpurple,
    # mediumseagreen, mediumslateblue, mediumspringgreen, mediumturquoise, mediumvioletred, midnightblue,
    # mintcream, mistyrose, moccasin, navajowhite, navy, oldlace, olive, olivedrab, orange, orangered,
    # orchid, palegoldenrod, palegreen, paleturquoise, palevioletred, papayawhip, peachpuff, peru, pink,
    # plum, powderblue, purple, red, rosybrown, royalblue, saddlebrown, salmon, sandybrown,
    # seagreen, seashell, sienna, silver, skyblue, slateblue, slategray, slategrey, snow, springgreen,
    # steelblue, tan, teal, thistle, tomato, turquoise, violet, wheat, white, whitesmoke, yellow, yellowgreen
    y_axis = 'Stub'
    y_format = '0.'

    def __init__(self, df=None, partition=None, partition_value=1, title=None, display=False):
        self.title = title
        self.display = display
        if isinstance(df, str):
            dt = Datatables()
            self.df = dt.get(df)
            if partition is not None:
                self.df = self.df[self.df[partition] == partition_value]
            self.metadata = dt.metadata
        elif isinstance(df, pd.DataFrame):
            self.df = df
        self.medians = None

    def finish(self, fig, figure, size=None):
        fig.update_layout(
            plot_bgcolor=self.background,
            xaxis=dict(linecolor='black', mirror=True, gridcolor=self.gridlines),
            yaxis=dict(linecolor='black', mirror=True, gridcolor=self.gridlines),
            font=dict(family='Arial', variant='small-caps',),
            legend_itemclick=False,
        )
        if size is not None:
            fig.update_layout(
                font_size=size,
            )
        if self.display:
            fig.show()
        plotly.offline.plot(fig, filename=str(self.image_path / f'{figure}.html'), auto_open=False)
        return self.image_path / f'{figure}.html'

    def plot(self, figure, unit='months', splitter=None, x_max=None, y_range=None, size=None, width=None, flip=False):

        if splitter is None:
            splitter = 'dummy'
            if 'dummy' not in self.df.columns:
                self.df['dummy'] = 1

        # get the unique values of the splitter
        if not flip:
            splits = sorted(self.df[splitter].unique())
        else:
            splits = reversed(sorted(self.df[splitter].unique()))

        # plot the curves
        return self._plot(figure, unit, splitter, splits, x_max, y_range, size, width)

    def callback(self, splitter, index, split):
        """
        Stub callback function for plotting curves for different models
        """
        t = np.array([])
        survival_prob = np.array([])
        return t, survival_prob

    def _plot(self, figure, unit, splitter, splits, x_max, y_range, size, width):

        # initialize each series
        self.medians = []
        p90 = []
        data = []
        sp_min = 1.0
        sp_max = 0.0
        if width is None:
            width = 2

        # build the traces
        max_y = 0
        for index, split in enumerate(splits):

            # get the survival curve data series
            t, y = self.callback(splitter, index, split)
            max_y = max(max_y, y.max())

            # attempt to calculate the median
            if '%' in self.y_format:
                if y.min() < 0.5:
                    y_list = y.tolist()
                    self.medians.append(t.iloc[y_list.index(y_list[min(range(len(y)),
                                                                       key=lambda i: abs(y_list[i] - 0.5))])])
                else:
                    self.medians.append(None)

            # calculate the 90th percentile (or as close as possible)
            cutoff = max(0.1, float(y.min()))
            try:
                p90.append(float(t[next(x[0] for x in enumerate(y) if x[1] <= cutoff)]))
            except KeyError:
                p90.append(float(t.iloc[-1]))

            # create a step function figure
            sp_min = min(sp_min, y.min())
            sp_max = max(sp_max, y.max())
            trace = {
                'x': t,
                'y': y,
                'line': {
                    'shape': 'hv',
                    'color': self.colors[index],
                    'width': width,

                },
                'mode': 'lines',
                'name': split if splitter != 'dummy' else '',
                'type': 'scatter',
            }
            data.append(trace)
        if abs(int(max_y + 0.5) - max_y) < 0.001:
            max_y = int(max_y + 0.5)
        fig = px.scatter()
        fig.add_traces(data)

        # annotate survival medians
        for median in self.medians:
            if median is not None:
                fig.add_hline(y=0.5, line_color=self.annotation_lines, line_width=5, layer='below',
                              annotation={'text': ' median survival'})
                fig.add_vline(x=median, line_color=self.annotation_lines, line_width=5, layer='below',
                              annotation={'text': f' {median:.1f}{'' if size is None else '<br>'} {unit}'})
                fig.update_annotations(font=dict(color=self.annotations))

        # set the ranges of the axes
        fig.update_xaxes(range=[0, x_max if x_max is not None else max(p90)])
        if hasattr(self, 'y_range') and self.y_range is not None:
            fig.update_yaxes(range=[*self.y_range])
        else:
            fig.update_yaxes(range=[max(0.0, sp_min - 0.10 * (sp_max - sp_min)), max_y])

        # add the axis labels
        fig.update_layout(
            xaxis=dict(title=dict(text=f'<b>Time since start of study (in {unit})</b>')),
            yaxis=dict(title=dict(text=f'<b>{self.y_axis}</b>'), tickformat=self.y_format),
            width=800,
            height=600,
        )
        if self.title is not None:
            fig.update_layout(title=self.title)

        # show and save the figure
        return self.finish(fig, figure, size)

if __name__ == '__main__':
    Plot()