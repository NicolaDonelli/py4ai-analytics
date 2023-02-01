"""Machine learning core module."""
from abc import ABC, abstractmethod, abstractproperty
from functools import total_ordering
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    overload,
    TypeVar,
    Generic,
)

import pandas as pd
from py4ai.data.model.core import BaseRange, DillSerialization
from py4ai.data.model.ml import TDatasetUtilsMixin, PandasDataset, PandasTimeIndexedDataset
from py4ai.data.model.text import Document, DocumentsUtilsMixin
from py4ai.core.logging import WithLogging
from typing_extensions import TypedDict

from py4ai.analytics.ml import ParamMixing


TDocumentsUtilsMixin = TypeVar("TDocumentsUtilsMixin", bound=DocumentsUtilsMixin)


class MetricType(TypedDict):
    """Metric type dictionary."""

    score: float
    loss: float


Numeric = Union[int, float]


class Enricher(ParamMixing, DillSerialization, ABC):
    """Abstract Applier class."""

    @abstractmethod
    def apply(self, obj: Any) -> Any:
        """
        Apply to an object.

        :param obj: original object

        :return: enriched object
        """
        ...


class Transformer(Enricher, ABC):
    """Abstract Transformer class."""

    # TODO [ND] wouldn't be better to make this an abstractproperty and delete HasParentEstimator?
    def estimator(self) -> Optional["Estimator"]:
        """
        Return the estimator used to train this estimator.

        :return: estimator
        """
        return None

    def transform(self, dataset: TDatasetUtilsMixin) -> TDatasetUtilsMixin:
        """
        Apply the transformation.

        :param dataset: dataset to apply the transformation
        :return: transformed dataset
        """
        return self.apply(dataset)

    def predict_proba(self, dataset):
        """Not implemented."""  # noqa: DAR
        raise NotImplementedError

    @property
    def hasPredictProba(self):
        """
        Return true if predict_proba method is implemented.

        :return: true if predict_proba method is implemented
        """
        return False


class Enhancer(Enricher, ABC):
    """Abstract class for a stage of enrichment of a document."""

    _type: Union[Document, TDocumentsUtilsMixin]

    @classmethod
    def get_type(cls) -> Union[Document, TDocumentsUtilsMixin]:
        """
        Return type of object processed by this enhancer.

        :return: type
        """
        return cls._type

    def _validate_input(self, documents: Union[Document, TDocumentsUtilsMixin]) -> None:
        """
        Validate the input (type checking).

        :param documents: object that must be validated
        :raises TypeError: raises if the type checking fails
        """
        if not isinstance(documents, self._type):
            raise TypeError(
                f"Input must be of type {self._type}. Type {type(documents)} found"
            )

    @overload
    def enhance(self, documents: Document) -> Document:
        ...

    @overload
    def enhance(self, documents: TDocumentsUtilsMixin) -> TDocumentsUtilsMixin:
        ...

    def enhance(
        self, documents: Union[Document, TDocumentsUtilsMixin]
    ) -> Union[Document, TDocumentsUtilsMixin]:
        """
        Transform an input document or set of documents to a document or set of documents with enriched property.

        :param documents: input
        :return: output
        """
        self._validate_input(documents)
        return self.apply(documents)


class Tokenizer(ABC):
    """Abstract tokenizer class."""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize."""
        ...  # noqa: DAR


class BaseFeatureProcessing(WithLogging, DillSerialization, ABC):
    """Abstract FeatureProcessing class."""

    def __init__(self, input_source):  # TODO [ND] define input_source type
        """
        Class instance initializer.

        :param input_source: input source
        """
        self.input_source = input_source

    @abstractmethod
    def dataset(self, range: Optional[List] = None) -> TDatasetUtilsMixin:
        """Not implemented."""
        raise NotImplementedError  # noqa: DAR

    @classmethod
    def with_input_source(
        cls, input
    ) -> "BaseFeatureProcessing":  # TODO [ND] define input type
        """
        Create an instance of this class with a given input.

        :param input: input
        :return: an instance of this class
        """
        return cls(input)


class FeatureProcessing(BaseFeatureProcessing, ABC):
    """
    Abstract FeatureProcessing class.

    It represents the connector between an input source and the downstream ML model.
    It provides the feature space and (optionally) the labels.
    """

    _output_type: type = PandasDataset

    @property
    def output_type(self) -> type:
        """
        Get output type.

        :raises ValueError: if output_type is not a class type that subclass PandasDataset

        :return: instance output type
        """
        if not issubclass(self._output_type, PandasDataset):
            raise ValueError(
                "output_type must be a class type that subclass PandasDataset"
            )
        return self._output_type

    @abstractmethod
    def feature_space(self, range: Optional[Any] = None) -> pd.DataFrame:
        """
        Create feature spaces from the given input source.

        :param range: range of data to consiedr
        :return: an object (e.g. pd.DataFrame) representing the feature space
        """
        raise NotImplementedError

    @abstractmethod
    def labels(
        self, range: Optional[Union[pd.Index, List[pd.Index]]] = None
    ) -> pd.DataFrame:
        """
        Create labels from the given input source.

        :param range: range of data to consiedr
        :return: an object (e.g. pd.DataFrame) representing the label(s)
        """
        raise NotImplementedError

    def dataset(
        self, range: Optional[Union[pd.Index, List[pd.Index]]] = None
    ) -> PandasDataset:
        """
        Create dataset of features and labels from the given input source.

        This is helpful in the data exploration phase where there is no need to split
        between a test and a train dataset.

        :param range: range of data to consiedr

        :raises IndexError: if feature or label space contains duplicated indexed entries

        :return: TDatasetUtilsMixin instance with features and labels
        """
        features = self.feature_space(range)
        if any(features.index.duplicated()):
            raise IndexError("Feature space contains duplicated indexed entries")

        labels = self.labels(range)
        if labels is not None and any(labels.index.duplicated()):
            raise IndexError("Label space contains duplicated indexed entries")

        return self.output_type(features, labels).intersection()

    def random_split(
        self,
        test_fraction: float = 0.2,
        random_state: Optional[int] = 42,
        subrange: Optional[Union[pd.Index, List[pd.Index], range]] = None,
    ) -> Tuple[PandasDataset, PandasDataset]:
        """
        Random split of the dataset.

        :param test_fraction: fraction of total samples to be used to produce the test dataset
        :param random_state: random seed; if None it does not randomize data before splitting
        :param subrange: List of indices (for the TDatasetUtilsMixin) to restrict the train/test splitting
        :return: tuple(TDatasetUtilsMixin, TDatasetUtilsMixin) with train and test dataset
        """
        full_dataset: PandasDataset = self.dataset(range=subrange).intersection()

        indices = pd.Series(full_dataset.features.index)

        if random_state is None:
            test = indices[: int(test_fraction * len(indices))]
        else:
            test = indices.sample(frac=test_fraction, random_state=random_state)

        train = list(set(indices).difference(test))

        return full_dataset.loc(train), full_dataset.loc(test)

    def split_by_indices(
        self,
        train_range: Union[pd.Index, List[pd.Index], range],
        test_range: Union[pd.Index, List[pd.Index], range],
    ) -> Tuple[PandasDataset, PandasDataset]:
        """
        Train/Test splitting based on list of indices values given as inputs.

        :param train_range: Iterable of indices to be used for training
        :param test_range: Iterable of indices to be used for testing
        :return: tuple(TDatasetUtilsMixin, TDatasetUtilsMixin) with train and test dataset
        """
        total_range = list(train_range) + list(test_range)

        full_dataset = self.dataset(total_range)

        return full_dataset.loc(train_range), full_dataset.loc(test_range)


class TimeSeriesFeatureProcessing(FeatureProcessing, ABC):
    """
    Abstract FeatureProcessingTimeSeriesSplit class.

    It represents the connector between an input source and the downstream
    ML model. It provides the feature space and (optionally) the labels as a PandasTimeIndexedDataset.
    It provides utilities to split the dataset in train/test sub-samples for the case of a time series type of
    data.
    """

    _output_type = PandasTimeIndexedDataset

    @abstractproperty
    def frequency(self) -> str:
        """
        Return the frequency of the samples that indices the DataFrame (for features and labels).

        Values can be for instance "D" for daily time series, "B" for business time series, "H" for hourly series, etc.
        The frequency should be used according to the range method of the py4ai.data.model.core.Range object

        :raises NotImplementedError: must be implemented by child classes
        """
        raise NotImplementedError

    @staticmethod
    def _validate_time_ranges(train_range: BaseRange, test_range: BaseRange):
        """
        Validate the time ranges.

        :param train_range: range of the training set
        :param test_range: range of the test set
        :raises ValueError: raises if the validation fails
        """
        if train_range.overlaps(test_range):
            raise ValueError(
                "The train and test ranges must be non-overlapping\n"
                f"Train: {str(train_range)}\n"
                f"Test: {str(test_range)}."
            )

    def split_by_time_range(
        self, train_range: BaseRange, test_range: BaseRange
    ) -> Tuple[PandasTimeIndexedDataset, PandasTimeIndexedDataset]:
        """
        Split data using the prescribed time-ranges, defined using the py4ai.data.model.core.BaseRange class.

        :param train_range: train range
        :param test_range: test range
        :return: the split dataset
        """
        self._validate_time_ranges(train_range, test_range)

        total_range = train_range + test_range
        dataset = self.dataset(total_range.range(self.frequency))

        return dataset.loc(train_range.range(self.frequency)), dataset.loc(
            test_range.range(self.frequency)
        )


class Optimizer(WithLogging, ABC, Generic[TDatasetUtilsMixin]):
    """Abstract base class for runners."""

    @abstractmethod
    def optimize(self, dataset: TDatasetUtilsMixin):
        """Optimize."""
        ...  # noqa: DAR


D = TypeVar("D")


class Splitter(WithLogging, Generic[D]):
    """Abstract base class for splitters."""

    @abstractmethod
    def nSplits(self, dataset: D) -> int:
        """
        Return the number of splits.

        :param dataset: input dataset
        :return: number of splits
        """
        raise NotImplementedError

    @abstractmethod
    def split(self, dataset: D) -> Iterable[Tuple[D, D]]:
        """
        Split feature space.

        :param dataset: input data

        :return: a couple of train and validation Datasets
        """
        raise NotImplementedError


class Estimator(ParamMixing, DillSerialization, ABC):
    """Abstract estimator class."""

    @abstractmethod
    def train(self, dataset: TDatasetUtilsMixin) -> "Transformer":
        """
        Train the estimator on a given dataset.

        :param dataset: input dataset
        :return: trained transformer
        """
        raise NotImplementedError


# TODO [ND] do we really need this class when we could simply make Transformer.estimator an abstractproperty?
class HasParentEstimator(object):
    """Has parent estimator."""

    @abstractmethod
    def estimator(self) -> Estimator:
        """Not implemented."""
        raise NotImplementedError


class Resampler(ParamMixing, ABC):
    """Abstract Resampler object."""

    @abstractmethod
    def resample(self, dataset: TDatasetUtilsMixin) -> TDatasetUtilsMixin:
        """
        Resample input dataset.

        :param dataset: input dataset
        :return: output dataset
        """
        raise NotImplementedError


@total_ordering
class Metric(ABC):
    """Base class for representing performance evaluations, e.g. Evaluator outputs."""

    @property
    @abstractmethod
    def value(self) -> float:
        """Not implemented."""
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        """
        Equality operator.

        :param other: other object to compare
        :raises ValueError: raises if the other object is of a different type than self
        :return: true if other == self
        """
        if not isinstance(other, Metric) or (type(self) != type(other)):
            raise ValueError(f"Cannot compare {type(self)} and {type(other)}")
        return self.value == other.value

    def __lt__(self, other: object) -> bool:
        """
        Less than operator.

        :param other: other object to compare
        :raises ValueError: raises if the other object is of a different type than self
        :return: true if other > self
        """
        if not isinstance(other, Metric) or (type(self) != type(other)):
            raise ValueError(f"Cannot compare {type(self)} and {type(other)}")
        return self.value < other.value


class Evaluator(WithLogging, ABC):
    """Evaluator abstract class."""

    @abstractmethod
    def evaluate(self, dataset: TDatasetUtilsMixin) -> Metric:
        """
        Evaluate metric on a given dataset.

        :param dataset: dataset to be evaluated
        :return: computed metric
        """
        raise NotImplementedError
