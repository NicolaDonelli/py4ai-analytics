"""Implementation of the TimeEvolvingSplitter class."""
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from py4ai.analytics.ml.core import Numeric, Splitter
from py4ai.data.model.ml import PandasTimeIndexedDataset
from pandas._libs.tslibs.nattype import NaTType
from pandas.core.tools.datetimes import DatetimeIndex, DatetimeScalar
from pandas.io.formats.printing import PrettyDict


class TimeEvolvingSplitter(Splitter[PandasTimeIndexedDataset]):
    """Create time consistent folds for cross validation."""

    def __init__(
        self,
        n_folds: Numeric = np.inf,
        train_ratio: Numeric = 0.9,
        min_periods_per_fold: int = 1,
        window: Optional[int] = None,
        valid_start: Optional[str] = None,
        g: Callable[[Any], Any] = lambda x: x,
    ) -> None:
        """
        Class instance initializer.

        :param train_ratio: ratio of data to be kept for training. Default = 0.9
        :param min_periods_per_fold: minimum number of time periods to include in each fold. Default = 1
        :param n_folds: maximum number of folds. Default = np.inf
        :param window: length of fixed window, in periods, to use for training set. Default = None
        :param valid_start: start date for the validation set. Default = None
        :param g: grouping function. Default = lambda x: x
        """
        self.min_periods_per_fold = min_periods_per_fold
        self.train_ratio = train_ratio
        self.valid_start = valid_start
        self.window = window
        self.n_folds = n_folds
        self.g = g

    @staticmethod
    def _get_first_period(
        _dict: Dict[str, DatetimeIndex],
        _start: Union[DatetimeIndex, pd.Series, DatetimeScalar, NaTType],
    ) -> str:
        """
        Get the key of the input dictionary associated to the first value that is greater or equal to a given timestamp.

        :param _dict: dictionary {"period": DatetimeIndex(["timestamps"])}
        :param _start: starting timestamp
        :return: key associated to the first timestamp equal or greater than _start
        """
        return sorted(filter(lambda x: (x[1] >= _start).any(), _dict.items()))[0][0]

    def _computeTimeframes(
        self, dataset: PandasTimeIndexedDataset
    ) -> Tuple[List[int], int, PrettyDict]:

        periods_dict = dataset.features.groupby(self.g).groups
        periods = sorted(periods_dict.keys())

        end = len(periods)

        start = (
            int(end * self.train_ratio)
            if (self.valid_start is None)
            else periods.index(
                self._get_first_period(
                    _dict=periods_dict, _start=pd.to_datetime(self.valid_start)
                )
            )
        )

        if (end - start) // self.n_folds == 0:
            self.logger.warning(
                "The number of periods is smaller than n_folds."
                "There will be %d folds with %d periods per fold."
                % (end - start, self.min_periods_per_fold)
            )
        elif (end - start) % self.n_folds != 0:
            self.logger.warning(
                "The number of periods is not multiple of n_folds = %d. "
                "n_folds + 1 folds will be generated with last one smaller than others"
                % self.n_folds
            )

        periods_per_fold = int(
            max((end - start) // self.n_folds, self.min_periods_per_fold)
        )
        starts = [int(x) for x in np.arange(start, end, periods_per_fold)]

        return starts, periods_per_fold, periods_dict

    def nSplits(self, dataset: PandasTimeIndexedDataset) -> int:
        """
        Return the number of splits.

        :param dataset: data
        :return: number of splits
        """
        return len(self._computeTimeframes(dataset)[0])

    def split(
        self, dataset: PandasTimeIndexedDataset
    ) -> Iterator[Tuple[PandasTimeIndexedDataset, PandasTimeIndexedDataset]]:
        """
        Generate folds for cross validation.

        :param dataset: dataset whose features are a pd.DataFrame with DateTimeIndex
        :yield: a couple of train and validation indices
        """
        starts, periods_per_fold, periods_dict = self._computeTimeframes(dataset)

        folds = len(starts)
        periods = sorted(periods_dict.keys())

        for ith, ind in enumerate(starts):
            self.logger.debug("Run %d out of %d" % (ith, folds))

            start = 0 if self.window is None else max(ind - self.window, 0)
            train_idx = [
                index for ii in periods[start:ind] for index in periods_dict[ii]
            ]
            test_idx = [
                index
                for ii in periods[ind : ind + periods_per_fold]
                for index in periods_dict[ii]
            ]
            yield dataset.loc(train_idx), dataset.loc(test_idx)

    def summary(self, dataset: PandasTimeIndexedDataset):
        """
        Print the dates for each fold.

        :param dataset: input data
        """
        for n, (Train, Valid) in enumerate(self.split(dataset)):
            trainPeriods = Train.features.groupby(self.g).groups
            validPeriods = Valid.features.groupby(self.g).groups

            trainMinPeriod = sorted(trainPeriods.keys())[0]
            trainMaxPeriod = sorted(trainPeriods.keys())[-1]

            trainNPeriods = len(trainPeriods)
            trainSamples = len(Train.index)

            validMinPeriod = sorted(validPeriods.keys())[0]
            validMaxPeriod = sorted(validPeriods.keys())[-1]

            validNPeriods = len(validPeriods)
            validSamples = len(Valid.index)

            self.logger.info(
                "Fold %d: --> Training: %d samples (%d periods)"
                % (n + 1, trainSamples, trainNPeriods)
            )
            self.logger.info(
                "         --> Start: %s  -  End: %s " % (trainMinPeriod, trainMaxPeriod)
            )
            self.logger.info(
                "         --> Test: %d samples (%d periods)"
                % (validSamples, validNPeriods)
            )
            self.logger.info(
                "         --> Start: %s  -  End: %s " % (validMinPeriod, validMaxPeriod)
            )
