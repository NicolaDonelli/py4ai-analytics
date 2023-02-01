"""Wrapper of the sklearn splitter."""
from typing import Union

from py4ai.data.model.ml import TDatasetUtilsMixin, PandasDataset

from py4ai.analytics.ml.core import Splitter
from py4ai.analytics.ml.wrapper.sklearn.wrapper import (
    GroupKFold,
    KFold,
    StratifiedKFold,
)


class SklearnSplitter(Splitter):
    """Splitter obtained by a sklearn class wrapped to return two sets of indices."""

    def __init__(self, skclass: Union[StratifiedKFold, GroupKFold, KFold]):
        """
        Initialize the class.

        :param skclass: wrapped sklearn class to be used for splitting
        """
        self.skclass = skclass

    def split(self, dataset: PandasDataset):
        """
        Generate train and validation datasets.

        :param dataset: input data

        :yield: couple of datasets
        """
        for iTrain, iValid in self.skclass.split(X=dataset.features, y=dataset.labels):
            yield dataset.loc(iTrain), dataset.loc(iValid)

    def nSplits(self, dataset: TDatasetUtilsMixin):
        """
        Get number of the split of a given Dataset.

        :param dataset: datatset to be splitted.
        :return: number of splits
        """
        return len(list(self.skclass.split(X=dataset.features, y=dataset.labels)))
