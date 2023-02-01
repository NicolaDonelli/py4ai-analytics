"""Function to plot stacked bar-charts using matplotlib."""
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_clustered_stacked(
    df,
    bars: list,
    index: str,
    stacked_bar: str,
    labels: Optional[List[str]] = None,
    title: str = "multiple stacked bar plot",
    H: str = "/",
    **kwargs
) -> Axes:
    """
    Given a df create a clustered stacked bar plot.

    :param df: dataframe
    :param bars: list of columns names used for having two or more bars side by side
    :param index: column name used for the x-axis
    :param stacked_bar: column name used for stacked bar
    :param labels: list of the names for bars legend
    :param title: string for the title of the plot
    :param H: hatch used for identification of the different columns
    :param kwargs: additional keyworded parameters are passed to the plot function of pd.DataFrame
    :return: image axis object
    """
    dfall = [df.pivot(index=index, columns=stacked_bar, values=b) for b in bars]

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall:  # for each data frame
        axe = df.plot(
            kind="bar",
            linewidth=0,
            stacked=True,
            ax=axe,
            legend=False,
            grid=False,
            **kwargs
        )

    hdl, lbl = axe.get_legend_handles_labels()  # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
        for j, pa in enumerate(hdl[i : i + n_col]):
            for rect in pa.patches:  # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col))
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.0)
    axe.set_xticklabels(df.index, rotation=0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n = [axe.bar(0, 0, color="gray", hatch=H * i) for i in range(n_df)]

    l1 = axe.legend(hdl[:n_col], lbl[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        bars = labels
    plt.legend(n, bars, loc=[1.01, 0.1])
    axe.add_artist(l1)
    return axe
