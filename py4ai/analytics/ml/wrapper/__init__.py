"""Wrapper module."""
from functools import wraps
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from py4ai.data.model.ml import TDatasetUtilsMixin, PandasDataset
from py4ai.core.utils.pandas import is_sparse
from deprecated import deprecated
from scipy.sparse import issparse, spmatrix
from sklearn.utils.multiclass import type_of_target

from py4ai.analytics.ml import ArrayLike

SklearnMethod = Callable[[object, Union[ArrayLike, spmatrix], Any, Any], ArrayLike]
ArrayFunc = Callable[[object, ArrayLike, Any, Any], ArrayLike]


def _toArray(X: ArrayLike):
    """
    Return an array from a series/dataframe.

    :param X: array-like object
    :return: array
    """
    if isinstance(X, pd.DataFrame) and is_sparse(X):
        return X.sparse.to_coo()
    elif isinstance(X, pd.DataFrame):
        return np.array(X)
    else:
        return X


def sparseWrapper(f: ArrayFunc) -> ArrayFunc:
    """
    Wrap methods to use sparse versions if possibile.

    :param f: method to wrap
    :return: wrapped method
    """

    @wraps(f)
    def wrapper(self, X, *args, **kwargs):
        return f(self, _toArray(X), *args, **kwargs)

    return wrapper


def seriesWrapper(
    method: SklearnMethod,
) -> Callable[[Union[ArrayLike, spmatrix], Any, Any], Union[np.ndarray, pd.DataFrame]]:
    """
    Wrap input method to return a pd.Series instead of a np.array.

    :param method: input method
    :return: wrapped method
    """

    @wraps(method)
    def wrapper(
        self: object, X: ArrayLike, *args: Any, **kwargs: Any
    ) -> Union[np.ndarray, pd.DataFrame]:
        # Run the function
        try:
            index = X.index
        except AttributeError:
            return method(self, X, *args, **kwargs)

        res = method(self, _toArray(X), *args, **kwargs)

        if issparse(res):
            raise ValueError("seriesWrapper cannot be used together with sparse arrays")

        return pd.Series(res, index=index).to_frame("pred")

    return wrapper


def dataframeWrapper(
    columns: Optional[List[str]], get_X_names: bool = False
) -> Callable[[SklearnMethod], SklearnMethod]:
    """
    Wrap input method to return a pd.DataFrame instead of a np.array.

    :param columns: optional list of colnames to use
    :param get_X_names: whether or not to take X column names if X.shape[1] == res.shape[1]
    :return: wrapped method
    """

    def real_decorator(method: SklearnMethod) -> SklearnMethod:
        @wraps(method)
        def wrapper(self: object, X: ArrayLike, *args: Any, **kwargs: Any) -> ArrayLike:
            # Run the function
            try:
                index = X.index
            except AttributeError:
                return method(self, X, *args, **kwargs)
            res = method(self, _toArray(X), *args, **kwargs)
            if columns is None:
                try:
                    input_dim = X.shape[1]
                except IndexError:
                    input_dim = 1
                if (input_dim == res.shape[1]) and get_X_names:
                    cols = X.columns
                else:
                    cols = np.arange(res.shape[1])
            else:
                cols = columns
            return (
                pd.DataFrame.sparse.from_spmatrix(res, index=index, columns=columns)
                if issparse(res)
                else pd.DataFrame(res, index=index, columns=cols)
            )

        return wrapper

    return real_decorator


def generatorWrapper(
    method: Callable[
        [Union[pd.DataFrame, pd.Series], Any, Any], Iterator[Tuple[Any, Any]]
    ]
) -> Callable[
    [Union[pd.DataFrame, pd.Series], Any, Any],
    Iterator[Tuple[Union[int, range], Union[int, range]]],
]:
    """
    Wrap input generator to return a generator with only (self, X) inputs.

    :param method: input method
    :return: wrapped method
    """

    def wrapper(self, X, *args, **kwargs):
        # Run the function
        try:
            index = X.index
        except AttributeError:
            # TODO [ND] what is the point of this exception if it's called the 'index' attribute that should be missing?
            index = range(len(X.index))
        for iTrain, iTest in method(self, X, *args, **kwargs):
            yield index[iTrain], index[iTest]

    return wrapper


def pandasDatasetWrapper(
    method: Callable[[TDatasetUtilsMixin, Any, Any], Tuple[ArrayLike, ArrayLike]]
) -> Callable[[TDatasetUtilsMixin, Any, Any], PandasDataset]:
    """
    Wrap input method to return a PandasDataset instead of a features, labels tuple.

    :param method: input method
    :return: wrapped method
    """

    @wraps(method)
    def wrapper(self, dataset, *args, **kwargs):
        features, labels = method(self, dataset, *args, **kwargs)
        return PandasDataset(features=features, labels=labels)

    return wrapper


@deprecated(
    "This wrapper is deprecated and will be removed in future releases of the package"
)
def generateEstimator(NativeClass: Any) -> Any:
    """
    Generate estimator.

    :param NativeClass: native class
    :return: estimator
    """

    class ClassOut(NativeClass):
        __class__ = NativeClass().__class__

        columns = None

        @seriesWrapper
        def predict(self, X):
            return super(ClassOut, self).predict(X)

    return ClassOut


@deprecated(
    "This wrapper is deprecated and will be removed in future releases of the package"
)
def generateTransformer(NativeClass: Any) -> Any:
    """
    Generate transformer.

    :param NativeClass: native class
    :return: transformer
    """

    class ClassOut(NativeClass):
        __class__ = NativeClass().__class__
        columns = None

        @dataframeWrapper(columns, get_X_names=True)
        def transform(self, X):
            return super(ClassOut, self).transform(X)

    return ClassOut


@deprecated(
    "This wrapper is deprecated and will be removed in future releases of the package"
)
def generateIterator(NativeClass: Any) -> Any:
    """
    Generate iterator.

    :param NativeClass: native class
    :return: iterator
    """

    class ClassOut(NativeClass):
        __class__ = NativeClass().__class__
        columns = None

        @generatorWrapper
        def split(self, X, *args, **kwargs):
            return super(ClassOut, self).split(X, *args, **kwargs)

    return ClassOut


@deprecated(
    "This wrapper is deprecated and will be removed in future releases of the package"
)
def labelsdiscretizerWrapper(
    method: Callable[
        [object, ArrayLike, Union[pd.Series, pd.DataFrame], Any, Any], ArrayLike
    ]
) -> Callable[[object, ArrayLike, Union[pd.Series, pd.DataFrame], Any, Any], ArrayLike]:
    """
    Discretize labels based on sign.

    :param method: fit-like method to be wrapped
    :return: wrapped method
    """

    @wraps(method)
    def wrapper(self, X, y, *args, **kwargs):
        if isinstance(y, pd.core.series.Series) or (
            isinstance(y, pd.core.frame.DataFrame) and y.shape[1] == 1
        ):
            pass
        else:
            raise ValueError(
                "Invalid target variable. It must be a pandas Series or a single-column pandas DataFrame."
            )
        if type_of_target(y) == "continuous":
            y = pd.Series(y.values.ravel())
            y = y.apply(lambda x: -1 if x < 0 else 1)
        return method(self, X, y, *args, **kwargs)

    return wrapper
