"""Utility functions used by the optimizer classes."""
import pandas as pd
from py4ai.data.model.ml import PandasDataset

from py4ai.analytics.ml.core import Estimator, Splitter
from py4ai.analytics.ml.wrapper.sklearn.splitter import SklearnSplitter
from py4ai.analytics.ml.wrapper.sklearn.wrapper import GroupKFold

_k_fold = SklearnSplitter(
    GroupKFold(n_splits=5, partition_key=lambda x: "%s-%s" % (x.year, x.month))
)


def kfold_prediction(
    estimator: Estimator, dataset: PandasDataset, strategy: Splitter = _k_fold
) -> pd.DataFrame:
    """
    Concatenate predictions obtained with cross validation process.

    :param estimator: estimator to validate
    :param dataset: train dataset
    :param strategy: splitting strategy

    :return: sorted (cross) validation predictions
    """
    agg = []
    for iTrain, iTest in strategy.split(dataset):
        pred = estimator.train(dataset.loc(iTrain)).transform(dataset.loc(iTest))
        agg.append(pred.getFeaturesAs('pandas'))
    return pd.concat(agg).sort_values()
