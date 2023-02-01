"""Scikit-learn estimator and transformer."""

from abc import ABC
from typing import Dict, List, Optional, Union

import pandas as pd
from py4ai.data.model.ml import TDatasetUtilsMixin, PandasDataset
from sklearn import clone
from sklearn.preprocessing import OneHotEncoder

from py4ai.analytics.ml import InputTypeSetter
from py4ai.analytics.ml.core import Estimator as CoreEstimator
from py4ai.analytics.ml.core import Transformer
from py4ai.analytics.ml.wrapper.sklearn import Skclass
from py4ai.analytics.ml.wrapper.sklearn.transformer import (
    KNNImputerTransformer,
    OneHotEncoderTransformer,
)
from py4ai.analytics.ml.wrapper.sklearn.wrapper import KNNImputer, MultiOutputClassifier


class GenericTransformer(Transformer, ABC):
    """Model obtained training a SklearnEstimator."""

    method: Optional[str] = None

    def __init__(self, skclass: Skclass, estimator: "Estimator") -> None:
        """
        Initialize the class.

        :param skclass: fitted scikit-learn class
        :param estimator: input estimator
        """
        self._estimator = estimator
        self.skclass = skclass

    def apply(self, dataset: TDatasetUtilsMixin) -> PandasDataset:
        """
        Predict the values of the labels given input features.

        :param dataset: input dataset

        :return: prediction
        :raises TypeError: dataset is not of type Dataset
        """
        if not isinstance(dataset, TDatasetUtilsMixin):
            raise TypeError("dataset must be of type Dataset")

        features = dataset.getFeaturesAs(self._estimator.input_type)

        return PandasDataset(
            getattr(self.skclass, self.method)(features),
            dataset.getLabelsAs(self.estimator.input_type),
        )

    @property
    def estimator(self) -> "Estimator":
        """
        Estimator generating the Model.

        :return: estimator generating the skclass
        """
        return self._estimator


class Model(GenericTransformer):
    """Model wrapper."""

    method = "predict"


class ProbabilisticClassifier(GenericTransformer):
    """Probabilistic Classifier wrapper."""

    method = "predict_proba_df"


class Transformer(GenericTransformer):
    """Transformer wrapper."""

    method = "transform"


class Estimator(CoreEstimator, InputTypeSetter):
    """Estimator obtained by a sklearn class wrapped to return a pd.DataFrame/Series."""

    types: List[type(GenericTransformer)] = [
        ProbabilisticClassifier,
        Model,
        Transformer,
    ]

    def __init__(self, skclass: Skclass, drop_na: bool = True):
        """
        Initialize the class.

        :param skclass: wrapped sklearn class to be trained
        :param drop_na: whether to drop nans
        :raises AttributeError: input skclass does not have predict nor transform attribute
        """
        self.skclass = skclass
        self.drop_na = drop_na

        if not (hasattr(self.skclass, "predict") or hasattr(self.skclass, "transform")):
            raise AttributeError(
                "Input skclass does not have predict nor transform attribute"
            )

    @staticmethod
    def validate_features(features: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Validate the features typo.

        :param features: data
        :return: data as a dataframe
        :raises TypeError: not a dataframe or series
        """
        if isinstance(features, pd.DataFrame):
            return features
        elif isinstance(features, pd.Series):
            return features.to_frame(features.name)
        else:
            raise TypeError(
                "Features is not of allowed type. Must be either a pandas DataFrame or Series"
            )

    @staticmethod
    def validate_labels(labels: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Validate the labels type.

        :param labels: data
        :return: data as a series
        :raises ValueError: if data have more than one column
        :raises TypeError: not a dataframe or series
        """
        if isinstance(labels, pd.DataFrame):
            if len(labels.columns) == 1:
                return labels[labels.columns[0]]
            else:
                raise ValueError(
                    "Sklearn Estimators only handles single columns labels"
                )
        elif isinstance(labels, pd.Series):
            return labels
        else:
            raise TypeError(
                "Labels is not of allowed type. Must be either a pandas DataFrame or Series"
            )

    def _train(self, dataset: PandasDataset) -> Skclass:
        """
        Train the skclass.

        :param dataset: Dataset
        :return: fitted skclass
        """
        if self.drop_na:
            cleaned = dataset.dropna().intersection()

            if len(cleaned) != len(dataset):
                self.logger.warning(
                    f"{type(self)}: Training Set has nan values which has been dropped from the training"
                )
        else:
            cleaned = dataset

        features, labels = (
            self.validate_features(cleaned.getFeaturesAs(self.input_type)),
            self.validate_labels(cleaned.getLabelsAs(self.input_type))
            if cleaned.labels is not None
            else None,
        )

        return (
            clone(self.skclass).fit(features, labels)
            if labels is not None
            else clone(self.skclass).fit(features)
        )

    def train(
        self, dataset: PandasDataset
    ) -> Union[ProbabilisticClassifier, Model, Transformer]:
        """
        Train the estimator with the given dataset.

        :param dataset: input dataset

        :return: prediction

        :raises TypeError: if dataset is not a PandasDataset
        """
        if not isinstance(dataset, PandasDataset):
            raise TypeError

        skclass = self._train(dataset)

        outputClass = [type for type in self.types if hasattr(skclass, type.method)][0]

        return outputClass(skclass, self)


class MultiOutputModel(Model):
    """Model for producing more than one prediction output. Used for multilabel classification and/or Neural Networks."""

    def __init__(
        self, skclass: Skclass, names: List[str], estimator: "MultiOutputEstimator"
    ):
        """
        Initialize the class.

        :param skclass: sklearn class to be wrapped in Model/Estimator abstraction
        :param names: Classes/Output columns names
        :param estimator: parent Estimator class
        """
        super(MultiOutputModel, self).__init__(skclass, estimator)
        self.names = names

    def transform(self, dataset: PandasDataset) -> PandasDataset:
        """
        Compute predictions based on input dataset.

        :param dataset: input dataset
        :return: output dataset with multilabel predictions and real data
        """
        transformedDataset = super(MultiOutputModel, self).transform(dataset)

        return PandasDataset(
            transformedDataset.features.rename(
                {ith: name for ith, name in enumerate(self.names)}, axis=1
            ),
            transformedDataset.labels,
        )


class MultiOutputEstimator(Estimator):
    """Multi output estimator."""

    def __init__(self, skclass: MultiOutputClassifier):
        """
        Initialize the class.

        :param skclass: a sklearn multi output classifier wrapper
        """
        super(MultiOutputEstimator, self).__init__(skclass)

    @staticmethod
    def validate_labels(labels: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that labels are pandas Dataframe.

        :param labels: data
        :return: data as a dataframe
        :raises TypeError: not a dataframe or series
        """
        if not isinstance(labels, pd.DataFrame):
            raise TypeError("Labels is not of allowed type. Must be a pandas DataFrame")
        return labels

    def train(self, dataset: PandasDataset) -> MultiOutputModel:
        """
        Train the model.

        :param dataset: data
        :return: trained model
        """
        model = super(MultiOutputEstimator, self).train(dataset)
        return MultiOutputModel(model.skclass, dataset.labels.columns, self)


class OneHotEncoderEstimator(CoreEstimator):
    """Estimator obtained by a sklearn.preprocessing.OneHotEncoder wrapped to return a pd.DataFrame/Series."""

    def __init__(
        self,
        encoder: OneHotEncoder,
        min_freq: Union[float, int, List[float], List[int]],
        columns: List[str],
    ):
        """
        Initialize class.

        :param encoder: wrapped OneHotEncoder
        :param min_freq: minimum frequency for a value to become a dummy category
        :param columns: columns to be Hot-Encoded
        """
        self.encoder = encoder
        self.min_freq = min_freq
        self.columns = columns

    def _getCategoryForColumn(
        self, features: PandasDataset, min_freq: Dict[str, float]
    ) -> List[int]:
        """
        Get the dummy categories for each column in the original dataset.

        :param features: dataset to be Hot-Encoded
        :param min_freq:  minimum frequency for a value to become a dummy category
        :return: dummy categories selected
        """
        s = features.value_counts()
        return sorted(s[s > min_freq].index)

    def train(self, dataset: PandasDataset) -> OneHotEncoderTransformer:
        """
        Train the OneHotEncoder on the input dataset.

        :param dataset: dataset to be trained
        :return: OneHotEconcoderTransformer
        :raises TypeError: value of min_freqis neither int nor float
        :raises ValueError: min_freq too high
        """
        self.logger.info("Starting train Dummy...")

        features = dataset.features[self.columns]
        _min_freq = {}
        for i in self.columns:

            try:
                if isinstance(self.min_freq[i], float):
                    _min_freq[i] = self.min_freq[i] * features.shape[0]
                elif isinstance(self.min_freq[i], int):
                    _min_freq[i] = self.min_freq[i]
                else:
                    raise TypeError("value of min_freq should be an int or float")

            except KeyError:
                _min_freq[i] = 0

        categories = [
            self._getCategoryForColumn(features[col], _min_freq[col])
            for col in self.columns
        ]

        if not all(categories):
            raise ValueError(
                "min_freq too high, there are no dummy categories for some or all of the columns"
            )

        encoder = clone(self.encoder).set_params(
            categories=categories, handle_unknown="ignore"
        )

        return OneHotEncoderTransformer(
            encoder.train(dataset.features[self.columns]), self.columns
        )


class KNNImputerEstimator(CoreEstimator):
    """Estimator obtained by a sklearn.impute.KNNImputer wrapped to return a pd.DataFrame/Series."""

    def __init__(self, knn: KNNImputer, columns: List[str]):
        """
        Initialize the class.

        :param knn: wrapped KNNImputer
        :param columns: columns to be imputed
        """
        self.knn = knn
        self.columns = columns

    def train(self, dataset: PandasDataset) -> KNNImputerTransformer:
        """
        Train the KNNImputer on the input dataset.

        :param dataset: dataset to be trained
        :return: KNNImputerTransformer
        """
        self.logger.info("Starting train KNN...")
        return KNNImputerTransformer(
            self.knn.fit(dataset.features[self.columns]), self.columns
        )
