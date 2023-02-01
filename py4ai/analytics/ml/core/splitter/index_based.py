"""Implementation of the IndexSplitter class."""

import pandas as pd
from typing import Iterator, Tuple, List
from py4ai.analytics.ml.core import Splitter
from py4ai.data.model.ml import PandasDataset


class IndexSplitter(Splitter[PandasDataset]):
    """Split based on index."""

    def __init__(self, train_index: List[pd.Index], valid_index: List[pd.Index]) -> None:
        """
        Class instance initializer.

        :param train_index: train data index
        :param valid_index: validation data index
        """
        self.train_index = train_index
        self.valid_index = valid_index

    def split(
        self, dataset: PandasDataset
    ) -> Iterator[Tuple[PandasDataset, PandasDataset]]:
        """
        Reproduce a split based on an index.

        :param dataset: PandasDataset or class with .loc method
        :yield: a couple of train and validation dataset
        """
        yield dataset.loc(self.train_index), dataset.loc(self.valid_index)

    def nSplits(self, dataset: PandasDataset) -> int:
        """
        Return number of splits.

        :param dataset: not used
        :return: always 1
        """
        return 1
