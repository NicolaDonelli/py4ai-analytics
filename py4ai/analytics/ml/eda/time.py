"""
Module containing only function lagSeries.

The function lags a DataFrame or Series of time series df by delta days.
"""
from datetime import timedelta
from typing import Union

import pandas as pd


def lagSeries(
    df: Union[pd.DataFrame, pd.Series], delta: timedelta
) -> Union[pd.DataFrame, pd.Series]:
    """
    Lags a DataFrame or Series of time series df by delta days.

    It effectively shifts all indices forward in time by lag_days

    :param df: input dataframe or series
    :param delta: timedelta to apply

    :return: shifted dataframe or series
    """
    lagged_index = df.index.map(lambda x: x + delta)
    result = df.copy()
    result.index = lagged_index

    return result
