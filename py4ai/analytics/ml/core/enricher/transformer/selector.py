"""Implementation of the FeatureSelector and LabelSelector classes."""
from typing import List, Optional, Union

import pandas as pd
from py4ai.data.model.ml import PandasDataset
from typeguard import typechecked

from py4ai.analytics.ml.core import Estimator, Numeric, Transformer


class FeatureSelector(Transformer):
    """Transformer that select features."""

    @property
    def estimator(self) -> Optional[Estimator]:
        """
        Return the estimator that was used to train this transformer.

        :return: estimator
        """
        return self._estimator

    def __init__(
        self,
        to_keep: Union[List[Union[str, Numeric]], pd.Index],
        estimator: Optional[Estimator] = None,
    ) -> None:
        """
        Class instance initializer.

        :param estimator: estimator
        :param to_keep: list of features to keep
        """
        self.to_keep = to_keep
        self._estimator = estimator

    @typechecked
    def apply(self, dataset: PandasDataset) -> PandasDataset:
        """
        Drop columns from  selected dataset features.

        :param dataset: Dataset instance with features and labels

        :return: Dataset without dropped features
        """
        x = dataset.features

        if type(x) == pd.DataFrame:
            x_ = x.copy()
        else:
            x_ = pd.DataFrame(x)

        return dataset.createObject(features=x_[self.to_keep], labels=dataset.labels)


class LabelSelector(Transformer):
    """Transformer that selects labels."""

    @property
    def estimator(self) -> Optional[Estimator]:
        """
        Return the estimator that was used to train this transformer.

        :return: estimator
        """
        return self._estimator

    def __init__(
        self, to_keep: List[str], estimator: Optional[Estimator] = None
    ) -> None:
        """
        Class instance initializer.

        :param estimator: estimator
        :param to_keep: list of labels to keep
        """
        self.to_keep = to_keep
        self._estimator = estimator

    @typechecked
    def apply(self, dataset: PandasDataset) -> PandasDataset:
        """
        Select labels from given dataset.

        :param dataset: Dataset instance with features and labels

        :return: Dataset without dropped features
        """
        return (
            dataset.createObject(features=dataset.features, labels=dataset.labels)
            if dataset.labels is not None
            else dataset
        )
