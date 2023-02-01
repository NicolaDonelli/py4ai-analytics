"""Implementation of the StackingEstimator class and related classes."""
from functools import reduce
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import pandas as pd
from py4ai.data.model.ml import TDatasetUtilsMixin, PandasDataset
from py4ai.core.utils.dict import union
from typeguard import typechecked

from py4ai.analytics.ml.core import Estimator, Splitter, Transformer
from py4ai.analytics.ml.core.pipeline.transformer import PipelineTransformer
from py4ai.analytics.ml.process.resumer import TopModelsFromRunner
from py4ai.analytics.ml.process.runner import RunnerResults


class CombinedTransformer(Transformer):
    """
    Transformer that combines the output of multiple Transformer in a single Dataset.

    Optionally, each transformer can operate on a subselection of a multi-index PandasDataset.
    """

    def __init__(
        self,
        transformers: List[Tuple[Optional[str], Optional[str], Transformer]],
        dropOthers: bool = False,
    ) -> None:
        """
        Class instance initializer.

        :param transformers: List of tuples with (NameOfFeature, NameOfLabel, Transformer)
        :param dropOthers: Whether other features not in NameOfFeature level should be dropped or retained
        """
        self.transformers = transformers
        self.dropOthers = dropOthers

    @staticmethod
    def _filterDatasetBy(
        dataset: PandasDataset, featuresName: Sequence[str]
    ) -> PandasDataset:
        """
        Filter dataset to retain only a subset of the features.

        :param dataset: input data
        :param featuresName: list of features to keep
        :return: new dataset
        """
        return (
            dataset
            if (featuresName is None)
            else PandasDataset(dataset.features[featuresName], dataset.labels)
        )

    @typechecked
    def apply(self, dataset: PandasDataset) -> PandasDataset:
        """
        Predict the values of the labels given input features.

        :param dataset: input feature
        :return: prediction
        """
        names = [tuple[0] for tuple in self.transformers]

        combined = pd.concat(
            union(
                {
                    labelsName: transformer.transform(
                        self._filterDatasetBy(dataset, featuresName)
                    ).features
                    for featuresName, labelsName, transformer in self.transformers
                },
                {
                    k: v
                    for k, v in dataset.features.items()
                    if not self.dropOthers and k not in names
                },
            ),
            axis=1,
        )

        return PandasDataset(combined, dataset.labels)

    def predict_proba(self, dataset: PandasDataset) -> PandasDataset:
        """
        Compute predictions on the given input dataset.

        :param dataset: Dataset
        :return: Predictions on the proviede dataset
        """
        aggregated = []

        for i, model in enumerate(self.models):
            aggregated.append(
                model.predict(dataset).rename({"pred": "model_%i" % i}, axis=1)
            )

        ensemble_dataset = dataset.createObject(
            features=pd.concat(aggregated, axis=1), labels=pd.Series()
        )
        self.logger.debug(
            "Columns: %s" % ",".join(map(str, ensemble_dataset.features.columns))
        )

        return self.ensembler.predict_proba(ensemble_dataset)


class StackingEstimator(Estimator):
    """Estimator that generates an ensembling models, that aggregates multiple modeling techniques."""

    ModelReferenceClasses = (Estimator, Transformer, str, TopModelsFromRunner)
    ModelReference = Union[Estimator, Transformer, str, TopModelsFromRunner]

    def __init__(
        self,
        estimators: List[Tuple[str, ModelReference]],
        ensembler: Union[Estimator, Transformer],
        folding: Optional[Splitter] = None,
    ) -> None:
        """
        Class instance initializer.

        :param estimators: List of Estimator to be ensembled
        :param ensembler: Estimator to merge models
        :param folding: Folding strategy to be used when evaluating ensembling feature space
        """
        self._validate_inputs(estimators, ensembler, folding)
        self.estimators = estimators
        self.ensembler = ensembler
        self.folding = folding

    @classmethod
    def _validate_inputs(
        cls,
        estimators: List[Tuple[str, ModelReference]],
        ensembler: Union[Estimator, Transformer],
        folding: Optional[Splitter],
    ) -> None:

        if not (isinstance(ensembler, Estimator) or isinstance(ensembler, Transformer)):
            raise TypeError(
                "The ensembler must be a class extending Transformer or Estimator."
            )

        if not all(
            [
                isinstance(estimator[1], cls.ModelReferenceClasses)
                for estimator in estimators
            ]
        ):
            raise TypeError(
                "All models must be classes extending Transformer or Estimator."
            )

        if isinstance(ensembler, Estimator) and any(
            [isinstance(estimator, Estimator) for estimator in estimators]
        ):
            if folding is None:
                raise ValueError(
                    "A folding object must be provided if training is required both for the ensembler and"
                    " any of the stacked models."
                )

    def _get_model(
        self, estimator: ModelReference, dataset: PandasDataset
    ) -> Iterator[Tuple[Transformer, TDatasetUtilsMixin]]:
        """
        Get the tuple (model, predictions on validation set) for a given estimator.

        :param estimator: provided estimator
        :param dataset: training dataset

        :yield: tuple (model, predictions)

        :raises ValueError: if folding is not set
        """
        if isinstance(estimator, Estimator):

            if isinstance(self.ensembler, Transformer):
                transformer = estimator.train(dataset)
                yield transformer, transformer.transform(dataset)
            else:
                if self.folding is None:
                    raise ValueError('Input folding parameter cannot be None if input ensemble is an Estimator')
                folds: List[PandasDataset] = [
                    self.logResult(f"Training fold {run}", "DEBUG")(
                        estimator.train(train).transform(test)
                    )
                    for run, (train, test) in enumerate(self.folding.split(dataset))
                ]
                yield estimator.train(dataset), reduce(
                    lambda d1, d2: d1.union(d2), folds[1:], folds[0]
                )

        elif isinstance(estimator, Transformer):
            yield estimator, estimator.transform(dataset)

        elif isinstance(estimator, str):
            runnerResults = RunnerResults(estimator)
            topRun = runnerResults.top_runs(1)[0]
            predictions = PandasDataset(
                runnerResults.get_predictions(topRun), dataset.labels
            ).intersection()
            yield runnerResults.get_model(topRun), predictions

        elif isinstance(estimator, TopModelsFromRunner):
            for key in estimator.model_results.top_runs(estimator.top):
                predictions = PandasDataset(
                    estimator.model_results.get_predictions(key), dataset.labels
                ).intersection()
                yield estimator.model_results.get_model(key), predictions

    def _clean_nans(self, dataset: PandasDataset) -> PandasDataset:
        """
        Drop NANs if any on the given dataset.

        :param dataset: input dataset
        :return: Dataset where rows with Nans are filtered out
        """
        dropped = dataset.dropna().intersection()

        self.logger.debug(
            f"Shape before and after NANs dropping: {len(dataset)} --> {len(dropped)}"
        )

        return dropped

    @staticmethod
    def compose(
        transformers: List[Tuple[str, Transformer]],
        ensembler: Transformer,
        estimator: Optional[Estimator] = None,
    ) -> PipelineTransformer:
        """
        Compose several transformers using the provided ensembler.

        :param transformers: list of name, transformer
        :param ensembler: transformer that combines several outputs
        :param estimator: estimator that was used to train the stacked model
        :return: stacking pipeline
        :raises TypeError: the provided ensembler is not a Transformer
        """
        if not (
            isinstance(ensembler, Transformer)
            and all(
                [
                    isinstance(transformer, Transformer)
                    for _, transformer in transformers
                ]
            )
        ):
            raise TypeError(
                "All models and the ensembler must be instances of classes extending Transformer."
            )

        stacking = CombinedTransformer(
            [(None, modelName, transformer) for modelName, transformer in transformers],
            dropOthers=True,
        )

        return PipelineTransformer(
            [
                ("stacking", stacking),
                ("ensembler", ensembler),
            ],
            estimator,
        )

    def train(self, dataset: PandasDataset) -> PipelineTransformer:
        """
        Train a StackingModel.

        :param dataset: PandasDataset, dataset
        :return: Trained Model
        """
        if isinstance(self.ensembler, Transformer) and all(
            [isinstance(estimator, Transformer) for estimator in self.estimators]
        ):
            return self.compose(self.estimators, self.ensembler, self)

        trans_tuple, agg_tuple = tuple(
            zip(
                *[
                    (
                        (modelName, transformer),
                        (modelName, predictions.getFeaturesAs("pandas")),
                    )
                    for modelName, estimator in self.estimators
                    for transformer, predictions in self._get_model(estimator, dataset)
                ]
            )
        )

        transformers = list(trans_tuple)
        aggregated = dict(agg_tuple)

        ensemble_dataset = self.logResult(
            lambda x: "Columns: %s" % ",".join(map(str, x.features.columns)), "DEBUG"
        )(PandasDataset(pd.concat(aggregated, axis=1), dataset.labels))

        ensembler = (
            self.ensembler.train(self._clean_nans(ensemble_dataset))
            if isinstance(self.ensembler, Estimator)
            else self.ensembler
        )

        return self.compose(transformers, ensembler, self)
