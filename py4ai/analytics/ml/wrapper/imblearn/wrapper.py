"""Wrap classes from Imbalanced learn."""
from typing import Tuple

from py4ai.data.model.ml import TDatasetUtilsMixin
from imblearn.over_sampling import SMOTE as NativeSMOTE
from imblearn.over_sampling import RandomOverSampler as NativeRandomOverSampler
from imblearn.under_sampling import RandomUnderSampler as NativeRandomUnderSampler

from py4ai.analytics.ml import ArrayLike
from py4ai.analytics.ml.core import Resampler
from py4ai.analytics.ml.wrapper import pandasDatasetWrapper


class RandomOverSampler(NativeRandomOverSampler, Resampler):
    """Wrapped NativeRandomOverSampler Resempler."""

    @pandasDatasetWrapper
    def resample(self, dataset: TDatasetUtilsMixin) -> Tuple[ArrayLike, ArrayLike]:
        """
        Resample the dataset.

        :param dataset: Dataset to resample
        :return: resampled Pandas Dataset
        """
        return super(RandomOverSampler, self).fit_resample(
            X=dataset.getFeaturesAs("pandas"), y=dataset.getLabelsAs("pandas")
        )


class SMOTE(NativeSMOTE, Resampler):
    """Wrapped NativeSMOTE Resempler."""

    @pandasDatasetWrapper
    def resample(self, dataset: TDatasetUtilsMixin) -> Tuple[ArrayLike, ArrayLike]:
        """
        Resample the dataset.

        :param dataset: Dataset to resample
        :return: resampled Pandas Dataset
        """
        return super(SMOTE, self).fit_resample(
            X=dataset.getFeaturesAs("pandas"), y=dataset.getLabelsAs("pandas")
        )


class RandomUnderSampler(NativeRandomUnderSampler, Resampler):
    """Wrapped NativeRandomUnderSampler Resempler."""

    @pandasDatasetWrapper
    def resample(self, dataset: TDatasetUtilsMixin) -> Tuple[ArrayLike, ArrayLike]:
        """
        Resample the dataset.

        :param dataset: Dataset to resample

        :return: resampled Pandas Dataset
        """
        return super(RandomUnderSampler, self).fit_resample(
            X=dataset.getFeaturesAs("pandas"), y=dataset.getLabelsAs("pandas")
        )
