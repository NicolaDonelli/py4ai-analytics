"""Vectorizer and VectorizerTranformer wrappers."""
from typing import Union

import pandas as pd
from py4ai.data.model.ml import TDatasetUtilsMixin, PandasDataset
from typeguard import typechecked

from py4ai.analytics.ml.wrapper.sklearn import clone
from py4ai.analytics.ml.wrapper.sklearn.estimator import Estimator, GenericTransformer
from py4ai.analytics.ml.wrapper.sklearn.wrapper import CountVectorizer, TfidfVectorizer


class Vectorizer(Estimator):
    """Train a transformer to convert a collection of documents to a matrix of features according to input skclass."""

    def __init__(
        self,
        skclass: Union[CountVectorizer, TfidfVectorizer],
        textField: str = "text",
        prefix: str = "bow",
    ) -> None:
        """
        Initialize the class.

        :param skclass: input class to train
        :param textField: name of the field in input dataframe containing text train the skclass on
        :param prefix: prefix for transformed fields, to be passed to trained transformer
        """
        super(Vectorizer, self).__init__(skclass)
        self.textField = textField
        self.prefix = prefix

    @typechecked
    def train(self, dataset: TDatasetUtilsMixin) -> "VectorizerTransformer":
        """
        Train estimator on given dataset.

        :param dataset: input data to train estimator on
        :return: trained transformer
        """
        features = self.validate_features(dataset.getFeaturesAs(self.input_type))[
            self.textField
        ]

        return VectorizerTransformer(
            clone(self.skclass).fit(features), self.textField, self.prefix, self
        )


class VectorizerTransformer(GenericTransformer):
    """Convert a collection of raw documents to a matrix of features according to input skclass."""

    def __init__(
        self,
        skclass: Union[CountVectorizer, TfidfVectorizer],
        textField: str,
        prefix: str,
        estimator: Vectorizer,
    ) -> None:
        """
        Initialize the class.

        :param skclass: input class to create features
        :param textField: name of the field in input dataframe containing text to be processed
        :param prefix: prefix for output fields
        :param estimator: estimator that generated this transformer
        """
        super(VectorizerTransformer, self).__init__(skclass, estimator)
        self.textField = textField
        self.prefix = prefix

    def generateFeatureName(self, k: str) -> str:
        """
        Generate name for vectorized columns.

        :param k: original name
        :return: new name
        """
        return f"{self.prefix}_{k}" if (self.prefix is not None) else k

    @typechecked
    def transform(self, dataset: TDatasetUtilsMixin) -> PandasDataset:
        """
        Vectorize given text column.

        :param dataset: input dataset
        :return: processed data
        """
        features = dataset.getFeaturesAs(self._estimator.input_type)

        vectorized = self.skclass.transform(features[self.textField]).rename(
            {
                v: self.generateFeatureName(k)
                for k, v in self.skclass.vocabulary_.items()
            },
            axis=1,
        )

        return (
            PandasDataset(
                pd.concat([features.drop(self.textField, axis=1), vectorized], axis=1),
                dataset.labels,
            )
            if len(set(features.columns).difference([self.textField])) > 0
            else PandasDataset(vectorized, dataset.labels)
        )
