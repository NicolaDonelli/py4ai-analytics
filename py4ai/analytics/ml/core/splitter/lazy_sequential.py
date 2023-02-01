"""Implementation of the LazySequentialSplitter class."""
from itertools import islice
from typing import Iterator, Tuple

from py4ai.analytics.ml.core import Splitter
from py4ai.data.model.core import IterGenerator
from py4ai.data.model.ml import LazyDataset


class LazySequentialSplitter(Splitter[LazyDataset]):
    """Sequentially generate folds for cross validation from lazy datasets."""

    def __init__(
        self,
        initial_train_size: int,
        folds_size: int,
        n_folds: int,
        fixed_train_size: bool = True,
    ) -> None:
        """
        Class instance initializer.

        :param initial_train_size: number of samples to be included in the train set
        :param n_folds: maximum number of validation folds
        :param folds_size: number of samples to be included in each validation fold
        :param fixed_train_size: whether to keep a fixed train size or make it grow each time including past validation
            folds
        """
        self.initial_train_size = initial_train_size
        self.fixed_train_size = fixed_train_size
        self.folds_size = folds_size
        self.n_folds = n_folds

    def nSplits(self, dataset: LazyDataset) -> int:
        """
        Return the number of splits.

        :param dataset: not used
        :return: number of folds
        """
        return self.n_folds

    def split(self, dataset: LazyDataset) -> Iterator[Tuple[LazyDataset, LazyDataset]]:
        """
        Generate folds for cross validation.

        :param dataset: input data
        :yield: a couple of LazyDataset
        """
        initial_train_size = self.initial_train_size
        fixed_train_size = self.fixed_train_size
        folds_size = self.folds_size

        for fold in range(self.n_folds):

            def train_generator():
                train_slice = (
                    islice(dataset, 0, initial_train_size + (fold * folds_size))
                    if not fixed_train_size
                    else islice(
                        dataset,
                        (fold * folds_size),
                        initial_train_size + (fold * folds_size),
                    )
                )
                for sample in train_slice:
                    yield sample

            def valid_generator():
                valid_slice = islice(
                    dataset,
                    initial_train_size + (fold * folds_size),
                    initial_train_size + ((fold + 1) * folds_size),
                )

                for sample in valid_slice:
                    yield sample

            train = LazyDataset(IterGenerator(train_generator))
            valid = LazyDataset(IterGenerator(valid_generator))

            if len(valid.getFeaturesAs("array")) != 0:
                yield train, valid
            else:
                break

    def summary(self, dataset: LazyDataset) -> None:
        """
        Print the dates for each fold.

        :param dataset: input data
        """
        for n, (Train, Valid) in enumerate(self.split(dataset)):
            self.logger.info(
                "Fold %d: --> Training: %d samples"
                % (n + 1, len(Train.getFeaturesAs("pandas")))
            )
            self.logger.info(
                "         --> Test: %d samples" % len(Valid.getFeaturesAs("pandas"))
            )
