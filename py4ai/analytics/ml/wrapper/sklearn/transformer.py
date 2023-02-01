"""
sklearn Transformers wrappers.

The module contains the following wrappers:
    - OneHotEncoderTransformer: wrapper of OneHotEncoder
    - KNNImputerTransformer: wrapper of KNNImputer.
"""
from typing import List, Optional

import numpy as np
import pandas as pd
from py4ai.data.model.ml import PandasDataset
from sklearn.feature_selection import (
    SelectFpr,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from typing_extensions import Literal

from py4ai.analytics.ml.core import Estimator
from py4ai.analytics.ml.core import Transformer as CoreTransformer
from py4ai.analytics.ml.core.enricher.transformer.discretizer import toMultiLevels
from py4ai.analytics.ml.core.enricher.transformer.selector import FeatureSelector
from py4ai.analytics.ml.wrapper.sklearn.wrapper import KNNImputer, OneHotEncoder

Task = Literal["Classification", "Regression"]


class SelectByMutualInfo(Estimator):
    """Select features based on scaled mutual information (for regression or classification tasks)."""

    __tasks = {
        "Classification": mutual_info_classif,
        "Regression": mutual_info_regression,
    }

    def __init__(
        self,
        task: Task,
        mi_thresh=0.01,
        y_thresh=None,
        discrete_features=False,
        n_neighbors=3,
    ) -> None:
        """
        Initialize the class.

        :param task: performed task.
            It can be 'Classification', 'Regression'
        :param mi_thresh: threshold on scaled mutual information (mutual information divided by its max).
            Drop columns if relative mutual information is smaller than thresh
        :param y_thresh: list of thresholds to discretize the target variable for classification tasks
        :param discrete_features: If bool, then determines whether to consider all features discrete or continuous.
            If array, then it should be either a boolean mask with shape (n_features,) or array with indices of discrete
            features. If 'auto', it is assigned to False for dense X and to True for sparse X.
        :param n_neighbors: Number of neighbors to use for MI estimation for continuous variables.
            Higher values reduce variance of the estimation, but could introduce a bias.
        """
        self.mi_thresh = mi_thresh
        self.y_thresh = y_thresh
        self.discrete_features = discrete_features
        self.n_neighbors = n_neighbors
        self.task = task

    def train(self, dataset: PandasDataset) -> FeatureSelector:
        """
        Train the feature selector.

        :param dataset: input data of the train
        :return: FeatureSelector Transformer
        """
        cleaned = dataset.dropna().intersection()

        if len(cleaned) != len(dataset):
            self.logger.warning(
                f"{type(self)}: Training Set has nan values which has been dropped from the training"
            )

        X, y = cleaned.features, cleaned.labels.squeeze()

        if self.task == "Classification" and self.y_thresh is not None:
            y = toMultiLevels(y, self.y_thresh)

        mi = self.__tasks[self.task](
            X,
            y,
            discrete_features=self.discrete_features,
            n_neighbors=self.n_neighbors,
            random_state=42,
        )
        if np.max(mi) != 0:
            mi /= np.max(mi) * 1.0
        scaled_mi = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        to_drop = list(scaled_mi[scaled_mi.values < self.mi_thresh].index)
        to_keep = list(set(X.columns).difference(to_drop))

        self.logger.info("Number of columns dropped by SelectByMI: %d" % len(to_drop))
        self.logger.debug(
            "SelectByMI dropped columns:\n%s" % ("\n".join(sorted(to_drop)))
        )

        return FeatureSelector(to_keep=to_keep, estimator=self)


class SelectByFtest(Estimator):
    """Select features based on F test (on the significance of linear regression parameters in case of regression task or an ANOVA in case of classification task). It drops the features with p-value less than the specified alpha."""

    __tasks = {"Classification": f_classif, "Regression": f_regression}

    def __init__(self, task: Task, y_thresh: Optional[List[float]] = None, alpha=0.05):
        """
        Initialize the class.

        :param task: performed task. It can be 'Classification' or 'Regression'.
        :param y_thresh: list of thresholds to discretize the target variable for classification tasks
        :param alpha: threshold on p-value computed from F statistic
        """
        self.alpha = alpha
        self.y_thresh = y_thresh
        self.task = task
        self.func = self.__tasks[self.task]

    def train(self, dataset: PandasDataset) -> FeatureSelector:
        """
        Train the feature selector.

        :param dataset: input data of the train
        :return: FeatureSelector Transformer
        """
        cleaned = dataset.dropna().intersection()

        if len(cleaned) != len(dataset):
            self.logger.warning(
                f"{type(self)}: Training Set has nan values which has been dropped from the training"
            )

        X, y = cleaned.features, cleaned.labels.squeeze()

        if self.task == "Classification" and self.y_thresh is not None:
            y = toMultiLevels(y, self.y_thresh)

        sfpr = SelectFpr(score_func=lambda x, y: self.func(x, y), alpha=self.alpha).fit(
            X, y
        )

        to_drop = list(
            set(X.columns).difference(X.columns[sfpr.get_support(indices=True)])
        )
        to_keep = list(set(X.columns).difference(to_drop))

        self.logger.info(
            "Number of columns dropped by SelectByFtest: %d" % len(to_drop)
        )
        self.logger.debug(
            "SelectByFtest dropped columns:\n%s" % ("\n".join(sorted(to_drop)))
        )

        return FeatureSelector(to_keep=to_keep, estimator=self)


class OneHotEncoderTransformer(CoreTransformer):
    """Model obtained training a OneHotEncoderEstimator."""

    def __init__(self, encoder: OneHotEncoder, columns: List[str]):
        """
        Initialize the class.

        :param encoder: wrapped OneHotEncoder
        :param columns: columns to be Hot-Encoded
        """
        self.encoder = encoder
        self.columns = columns

    def apply(self, dataset: PandasDataset) -> PandasDataset:
        """
        Apply the transformer on a dataset.

        :param dataset: pandas dataset to be transformed
        :return: transformed pandas dataset
        """
        return PandasDataset(
            self.encoder.transform(dataset.features[self.columns]).join(
                dataset.features.drop(self.columns, axis=1)
            ),
            dataset.labels,
        )


class KNNImputerTransformer(CoreTransformer):
    """Model obtained training a KNNImputerEstimator."""

    def __init__(self, model: KNNImputer, columns: List[str]):
        """
        Initialize the class.

        :param model: wrapped KNNImputer
        :param columns: columns to be imputed
        """
        self.model = model
        self.columns = columns

    def apply(self, dataset: PandasDataset) -> PandasDataset:
        """
        Apply the transformer on a dataset.

        :param dataset: pandas dataset to be transformed
        :return: transformed pandas dataset
        """
        df_transformed = self.model.transform(dataset.features[self.columns])
        return PandasDataset(
            df_transformed.join(dataset.features.drop(self.columns, axis=1))[
                dataset.features.columns
            ],
            dataset.labels,
        )
