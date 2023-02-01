"""core module of Scikit-learn wrappers."""
from typing import Any, Callable, Iterator, Optional, Tuple, Union

import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from scipy.sparse import spmatrix
from sklearn.cluster import KMeans as NativeKMeans
from sklearn.decomposition import PCA as NativePCA
from sklearn.ensemble import AdaBoostClassifier as NativeAdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor as NativeAdaBoostRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier as NativeGradientBoostingClassifier,
)
from sklearn.ensemble import (
    GradientBoostingRegressor as NativeGradientBoostingRegressor,
)
from sklearn.ensemble import RandomForestClassifier as NativeRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as NativeRandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer as NativeCountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as NativeTfidfVectorizer
from sklearn.impute import KNNImputer as NativeKNNImputer
from sklearn.linear_model import LogisticRegression as NativeLogisticRegression
from sklearn.model_selection import GroupKFold as NativeGroupKFold
from sklearn.model_selection import KFold as NativeKFold
from sklearn.model_selection import StratifiedKFold as NativeStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier as NativeOneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier as NativeMultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB as NativeMultinomialNB
from sklearn.neighbors import KNeighborsClassifier as NativeKNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor as NativeKNeighborsRegressor
from sklearn.neural_network import MLPClassifier as NativeMLPClassifier
from sklearn.neural_network import MLPRegressor as NativeMLPRegressor
from sklearn.pipeline import Pipeline as NativePipeline
from sklearn.preprocessing import OneHotEncoder as NativeOneHotEncoder
from sklearn.preprocessing import RobustScaler as NativeRobustScaler
from sklearn.preprocessing import StandardScaler as NativeStandardScaler
from sklearn.svm import SVC as NativeSVC
from sklearn.svm import SVR as NativeSVR
from sklearn.svm import LinearSVC as NativeLinearSVC

from py4ai.analytics.ml import ArrayLike
from py4ai.analytics.ml.wrapper import (
    dataframeWrapper,
    generatorWrapper,
    seriesWrapper,
    sparseWrapper,
)
from py4ai.analytics.ml.wrapper.sklearn import Skclass

SplitGenerator = Iterator[Tuple[ndarray, ndarray]]


class SparseDataFrameFitter(object):
    """Class to fit dataframe using sparse methods if possible."""

    @sparseWrapper
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: Optional[ArrayLike] = None,
    ) -> object:
        """
        Fit input data with sparse methods if possible.

        :param X: input features
        :param y: input labels
        :param sample_weight: input sample weights
        :return: fitted object
        """
        return super().fit(X, y, sample_weight=sample_weight)


class GroupKFold(NativeGroupKFold):
    """Wrap scikit-learn GroupKFold."""

    def __init__(
        self,
        partition_key: Callable[[pd.Index], Any] = lambda x: x.date(),
        n_splits: int = 3,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the class.

        :param partition_key: function used to get the group. The function will be applied to the dataframe indexes
        :param n_splits: number of splits
        :param shuffle: If True shaffle the rows
        :param random_state: random state
        """
        self.partition_key = partition_key
        super(NativeGroupKFold, self).__init__(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    @generatorWrapper
    def split(self, X: DataFrame, y: Optional[DataFrame] = None) -> SplitGenerator:
        """
        Split X and y following scikit-learn GroupKFold class.

        :param X: input features
        :param y: input labels
        :return: split generator
        """
        return super(NativeGroupKFold, self).split(
            X, groups=list(map(self.partition_key, X.index))
        )


class KFold(NativeKFold):
    """Wrap scikit-learn KFold."""

    @generatorWrapper
    def split(self, X: DataFrame, y: Optional[DataFrame] = None) -> SplitGenerator:
        """
        Split X and y following scikit-learn KFold class.

        :param X: input features
        :param y: input labels
        :return: split generator
        """
        return super(NativeKFold, self).split(X)


class StratifiedKFold(NativeStratifiedKFold):
    """Wrap scikit-learn StratifiedKFold."""

    @generatorWrapper
    def split(self, X: DataFrame, y: Optional[DataFrame]) -> SplitGenerator:
        """
        Split X and y following scikit-learn StratifiedKFold class.

        :param X: input features
        :param y: input labels
        :return: split generator
        """
        return super(NativeStratifiedKFold, self).split(X, y)


class RobustScaler(NativeRobustScaler, Skclass):
    """Wrap scikit-learn RobustScaler."""

    @dataframeWrapper(None, get_X_names=True)
    def transform(self, X: ArrayLike) -> Union[ndarray, spmatrix]:
        """
        Tranform the input X.

        :param X: input features
        :return: transformed features
        """
        return super(RobustScaler, self).transform(X)


class KMeans(NativeKMeans, Skclass):
    """Wrap scikit-learn KMeans."""

    @seriesWrapper
    def predict(self, X: ArrayLike) -> ndarray:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super(KMeans, self).predict(X)

    @seriesWrapper
    def fit_predict(self, X: ArrayLike, y: Optional[DataFrame] = None) -> ndarray:
        """
        Fit and Predict target for X.

        :param X: input features
        :param y: input labels
        :return: predictions
        """
        return super(KMeans, self).fit_predict(X, y=y)


class StandardScaler(NativeStandardScaler, Skclass):
    """Wrap scikit-learn StandardScaler."""

    @dataframeWrapper(None, get_X_names=True)
    def transform(self, X: ArrayLike) -> Union[ndarray, spmatrix]:
        """
        Tranform the input X.

        :param X: input features
        :return: transformed features:
        """
        return super(StandardScaler, self).transform(X)


class Pipeline(NativePipeline, Skclass):
    """Wrap scikit-learn Pipeline."""

    @seriesWrapper
    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict target for X.

        :param X: input features
        :return: transformed features
        """
        return super(Pipeline, self).predict(X)

    @dataframeWrapper(None)
    def predict_proba_df(self, X: ArrayLike) -> ArrayLike:
        """
        Predict proba for X.

        :param X: input features
        :return: predictions
        """
        return super(Pipeline, self).predict_proba(X)


class PCA(NativePCA, Skclass):
    """Wrap scikit-learn PCA."""

    @dataframeWrapper(None)
    def transform(self, X: ArrayLike) -> ArrayLike:
        """
        Tranform the input X.

        :param X: input features
        :return: tranformed features
        """
        return super(PCA, self).transform(X)


class CountVectorizer(NativeCountVectorizer, Skclass):
    """Wrap scikit-learn CountVectorizer."""

    @dataframeWrapper(None)
    def transform(self, X: ArrayLike) -> ArrayLike:
        """
        Tranform the input X.

        :param X: input features
        :return: transformed features
        """
        return super(CountVectorizer, self).transform(X)


class TfidfVectorizer(NativeTfidfVectorizer, Skclass):
    """Wrap scikit-learn TfidfVectorizer."""

    @dataframeWrapper(None)
    def transform(self, X: ArrayLike) -> ArrayLike:
        """
        Tranform the input X.

        :param X: input features
        :return: transformed features
        """
        return super(TfidfVectorizer, self).transform(X)


# Regressors
class SVR(NativeSVR, Skclass):
    """Wrap scikit-learn SVR."""

    @seriesWrapper
    def predict(self, X: ArrayLike) -> ndarray:
        """
        Predict target for X.

        :param X: input features
        :return: transformed features
        """
        return super(SVR, self).predict(X)


class GradientBoostingRegressor(NativeGradientBoostingRegressor, Skclass):
    """Wrap scikit-learn GradientBoostingRegressor."""

    @seriesWrapper
    def predict(self, X: ArrayLike) -> ndarray:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super(GradientBoostingRegressor, self).predict(X)


class AdaBoostRegressor(NativeAdaBoostRegressor, Skclass):
    """Wrap scikit-learn AdaBoostRegressor."""

    @seriesWrapper
    def predict(self, X: ArrayLike) -> ndarray:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super(AdaBoostRegressor, self).predict(X)


class KNeighborsRegressor(NativeKNeighborsRegressor, Skclass):
    """Wrap scikit-learn KNeighborsRegressor."""

    @seriesWrapper
    def predict(self, X: ArrayLike) -> ndarray:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super(KNeighborsRegressor, self).predict(X)


class MLPRegressor(NativeMLPRegressor, Skclass):
    """Wrap scikit-learn MLPRegressor."""

    @seriesWrapper
    def predict(self, X: ArrayLike) -> ndarray:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super(MLPRegressor, self).predict(X)


class RandomForestRegressor(NativeRandomForestRegressor, Skclass):
    """Wrap scikit-learn RandomForestRegressor."""

    @seriesWrapper
    def predict(self, X: ArrayLike) -> ndarray:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super(RandomForestRegressor, self).predict(X)


class MLPClassifier(NativeMLPClassifier, Skclass):
    """Wrap scikit-learn MLPClassifier."""

    @seriesWrapper
    def predict(self, X: ArrayLike) -> ndarray:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super(MLPClassifier, self).predict(X)

    @dataframeWrapper(None)
    def predict_proba_df(self, X: ArrayLike) -> ndarray:
        """
        Predict proba for X.

        :param X: input features
        :return: predictions
        """
        return super(MLPClassifier, self).predict_proba(X)


class GradientBoostingClassifier(NativeGradientBoostingClassifier, Skclass):
    """Wrap scikit-learn GradientBoostingClassifier."""

    @seriesWrapper
    def predict(self, X: ArrayLike) -> ndarray:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super(GradientBoostingClassifier, self).predict(X)

    @dataframeWrapper(None)
    def predict_proba_df(self, X: ArrayLike) -> ndarray:
        """
        Predict proba for X.

        :param X: input features
        :return: predictions
        """
        return super(GradientBoostingClassifier, self).predict_proba(X)


class AdaBoostClassifier(NativeAdaBoostClassifier, Skclass):
    """Wrap scikit-learn AdaBoostClassifier."""

    @seriesWrapper
    def predict(self, X: ArrayLike) -> ndarray:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super(AdaBoostClassifier, self).predict(X)

    @dataframeWrapper(None)
    def predict_proba_df(self, X: ArrayLike) -> ndarray:
        """
        Predict proba for X.

        :param X: input features
        :return: predictions
        """
        return super(AdaBoostClassifier, self).predict_proba(X)


class OneVsRestClassifier(NativeOneVsRestClassifier, Skclass):
    """Wrap scikit-learn OneVsRestClassifier."""

    @seriesWrapper
    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super(OneVsRestClassifier, self).predict(X)

    @dataframeWrapper(None)
    def predict_proba_df(self, X: ArrayLike) -> ArrayLike:
        """
        Predict proba for X.

        :param X: input features
        :return: predictions
        """
        return super(OneVsRestClassifier, self).predict_proba(X)


class RandomForestClassifier(NativeRandomForestClassifier, Skclass):
    """Wrap scikit-learn RandomForestClassifier."""

    @seriesWrapper
    def predict(self, X: Union[ArrayLike, spmatrix]) -> ndarray:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super(RandomForestClassifier, self).predict(X)

    @dataframeWrapper(None)
    def predict_proba_df(self, X: Union[ArrayLike, spmatrix]) -> ndarray:
        """
        Predict proba for X.

        :param X: input features
        :return: predictions
        """
        return super(RandomForestClassifier, self).predict_proba(X)


class KNeighborsClassifier(NativeKNeighborsClassifier, Skclass):
    """Wrap scikit-learn KNeighborsClassifier."""

    @seriesWrapper
    def predict(self, X: ArrayLike) -> ndarray:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super(KNeighborsClassifier, self).predict(X)

    @dataframeWrapper(None)
    def predict_proba_df(self, X: ArrayLike) -> ndarray:
        """
        Predict proba for X.

        :param X: input features
        :return: predictions
        """
        return super(KNeighborsClassifier, self).predict_proba(X)


class LogisticRegression(NativeLogisticRegression, Skclass):
    """Wrap scikit-learn LogisticRegression."""

    @seriesWrapper
    def predict(self, X: Union[ArrayLike, spmatrix]) -> ndarray:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super(LogisticRegression, self).predict(X)

    @dataframeWrapper(None)
    def predict_proba_df(self, X: ArrayLike) -> ndarray:
        """
        Predict proba for X.

        :param X: input features
        :return: predictions
        """
        return super(LogisticRegression, self).predict_proba(X)


class SVC(NativeSVC, Skclass):
    """Wrap scikit-learn SVC."""

    @seriesWrapper
    def predict(self, X: Union[ArrayLike, spmatrix]) -> ndarray:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super(SVC, self).predict(X)

    @dataframeWrapper(None)
    def predict_proba_df(self, X: ArrayLike) -> ndarray:
        """
        Predict proba for X.

        :param X: input features
        :return: predictions
        """
        return super(SVC, self).predict_proba(X)


class LinearSVC(NativeLinearSVC, Skclass):
    """Wrap scikit-learn LinearSVC."""

    @seriesWrapper
    def predict(self, X: Union[ArrayLike, spmatrix]) -> ndarray:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super(LinearSVC, self).predict(X)


class MultiOutputClassifier(
    SparseDataFrameFitter, NativeMultiOutputClassifier, Skclass
):
    """Wrap scikit-learn MultiOutputClassifier."""

    @dataframeWrapper(None)
    def predict(self, X: Union[ArrayLike, spmatrix]) -> Union[ArrayLike, spmatrix]:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super().predict(X)


class MultinomialNB(SparseDataFrameFitter, NativeMultinomialNB, Skclass):
    """Wrap scikit-learn MultinomialNB."""

    @seriesWrapper
    def predict(self, X: ArrayLike) -> ndarray:
        """
        Predict target for X.

        :param X: input features
        :return: predictions
        """
        return super().predict(X)


class KNNImputer(NativeKNNImputer, Skclass):
    """Wrap scikit-learn KNNImputer."""

    @dataframeWrapper(None, get_X_names=True)
    def transform(self, X: ArrayLike) -> ndarray:
        """
        Transform the input.

        :param X: input features
        :return: predictions
        """
        return super(KNNImputer, self).transform(X)


class OneHotEncoder(NativeOneHotEncoder):
    """Wrap scikit-learn OneHotEncoder."""

    def train(self, X: ArrayLike):
        """
        Train the encoder.

        :param X: ArrayLike input of the train
        :return: one hot encoder transformer
        """
        columns = (
            X.columns if isinstance(X, (pd.DataFrame, pd.Series)) else range(X.shape[1])
        )

        transformer = super(OneHotEncoder, self).fit(X)
        self.new_columns = [
            f"{col}_{level}"
            for icol, col in enumerate(columns)
            for level in transformer.categories_[icol]
        ]
        return transformer

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        """
        Transform the array like input in a pandas dataframe.

        :param X: ArrayLike input to be transformed
        :return: pandas dataframe
        """
        index = (
            X.index if isinstance(X, (pd.DataFrame, pd.Series)) else range(X.shape[0])
        )

        return pd.DataFrame.sparse.from_spmatrix(
            super(OneHotEncoder, self).transform(X),
            columns=self.new_columns,
            index=index,
        )
