"""Implementation of the PipelineEstimator and PipelineTransformer class."""

from functools import reduce
from typing import Any, List

from py4ai.analytics.ml.core import Estimator, Resampler
from py4ai.analytics.ml.core import HasParentEstimator, Transformer
from py4ai.analytics.ml.core.pipeline import BaseComposition
from py4ai.analytics.ml.core.pipeline import Pipeline, Step
from py4ai.data.model.ml import TDatasetUtilsMixin


class PipelineTransformer(Pipeline, Transformer, HasParentEstimator):
    """Trained PipelineEstimator."""

    def __init__(self, steps: List[Step], estimator: "PipelineEstimator") -> None:
        """
        Class instance initializer.

        :param steps: list of couples (name, Transformer) constituting the steps of the pipeline
        :param estimator: estimator that generates the model when trained
        """
        super(PipelineTransformer, self).__init__(steps)
        self._estimator = estimator

    def _validate_steps(self) -> None:
        """
        Check if all steps are Transformers.

        :raises TypeError: if any step is not of Transformer type
        """
        if any([not isinstance(stage[1], Transformer) for stage in self.steps]):
            raise TypeError("Each step of the pipeline must be a Transformer")

    @property
    def estimator(self) -> "PipelineEstimator":
        """
        Estimator from which the model is trained.

        :return: estimator
        """
        return self._estimator

    def apply(self, dataset: TDatasetUtilsMixin) -> Any:
        """
        Predict the values of the labels given input features.

        :param dataset: input feature
        :return: transformed dataset
        """
        return reduce(
            lambda intermediateDataset, transformer: transformer.transform(
                intermediateDataset
            ),
            list(zip(*self.steps))[1],
            dataset,
        )

    @property
    def hasPredictProba(self) -> bool:
        """
        Return true if the last step has a predict_proba method.

        :return: true if the last step has a predict_proba method
        """
        return hasattr(self.steps[-1][1], "predict_proba")

    def predict_proba(self, dataset: TDatasetUtilsMixin) -> Any:
        """
        Predict the probabilities of the labels given input features.

        :param dataset: input feature
        :return: predicted probabilities
        """
        # TODO :rtype: per il momento non e' ancora codificata la struttura delle previsioni

        _, models = zip(*self.steps)
        return models[-1].transform(
            reduce(lambda x, y: y.transform(x), models[:-1], dataset)
        )


class PipelineEstimator(Pipeline, BaseComposition):
    """Estimator constituted by a sequence of Estimators, Transformers or Resamplers finishing with an Estimator."""

    def __init__(self, steps: List[Step]) -> None:
        """
        Class instance initializer.

        :param steps: list of couples (name, object) constituting the steps of the pipeline
        """
        super(PipelineEstimator, self).__init__(steps)

    def _validate_steps(self) -> None:
        """
        Validate input steps.

        :raises ValueError: if intermediate steps are not of type Estimator, Transformer or Resampler
        :raises TypeError: if last step is not an Estimator or Transformer
        """
        _, estimators = zip(*self.steps)
        if any(
            [
                not (
                    isinstance(stage, Transformer)
                    or isinstance(stage, Estimator)
                    or isinstance(stage, Resampler)
                )
                for stage in estimators[:-1]
            ]
        ):
            raise ValueError(
                "Intermediate steps of pipeline must be of type Estimator, Transformer or Resampler"
            )

        if not (
            isinstance(estimators[-1], Estimator)
            or isinstance(estimators[-1], Transformer)
        ):
            raise TypeError(
                "Last step of pipeline must be of type Estimator or Transformer"
            )

    def clone(self) -> "PipelineEstimator":
        """
        Clone pipeline.

        :return: cloned pipeline estimator
        """
        return PipelineEstimator(
            [
                (step[0], step[1].clone()) if isinstance(step[1], Estimator) else step
                for step in self.steps
            ]
        )

    def insert(self, pos: int, step: Step) -> "PipelineEstimator":
        """
        Create pipeline estimator with updated steps list.

        :param pos: position in steps list to insert the new step
        :param step: tuple to add

        :return: newly initialized pipeline estimator
        """
        return PipelineEstimator(steps=self.steps[:pos] + [step] + self.steps[pos:])

    def __find_ref_id__(self, name: str) -> int:
        """Find the step with a given name.

        :param name: name of the step to be searched
        :return: the index of the step with the given name
        """
        names, _ = zip(*self.steps)
        return names.index(name)

    def insert_before_name(self, name: str, step: Step) -> "PipelineEstimator":
        """
        Create pipeline estimator with updated steps list inserting new step before given reference step.

        :param name: name of the sep to be used as reference
        :param step: tuple to add
        :return: newly initialized pipeline estimator
        """
        idx = self.__find_ref_id__(name)
        return PipelineEstimator(steps=self.steps[:idx] + [step] + self.steps[idx:])

    def insert_after_name(self, name: str, step: Step) -> "PipelineEstimator":
        """
        Create pipeline estimator with updated steps list inserting new step before given reference step.

        :param name: name of the sep to be used as reference
        :param step: tuple to add
        :return: newly initialized pipeline estimator
        """
        idx = self.__find_ref_id__(name)
        return PipelineEstimator(
            steps=self.steps[: (idx + 1)] + [step] + self.steps[(idx + 1) :]
        )

    @staticmethod
    def _append_step(steps: List[Step], step: Step) -> List[Step]:
        """
        Extend steps list with appending given step.

        :param steps: list of couples (name, object) constituting the steps of the pipeline
        :param step: tuple (name, object) to append to steps
        :return: updated list
        """
        return steps + [step]

    @staticmethod
    def _prepend_step(steps: List[Step], step: Step) -> List[Step]:
        """
        Extend steps list with pre-pending given step.

        :param steps: list of couples (name, object) constituting the steps of the pipeline
        :param step: tuple (name, object) to prepend to steps
        :return: updated list
        """
        return [step] + steps

    def append(self, step: Step) -> "PipelineEstimator":
        """
        Create pipeline estimator with steps list updated appending input step.

        :param step: tuple of the form ("name", step)
        :return: updated pipeline estimator
        """
        return PipelineEstimator(steps=self._append_step(self.steps, step))

    def prepend(self, step: Step) -> "PipelineEstimator":
        """
        Create pipeline estimator with steps list updated prepending input step.

        :param step: tuple of the form ("name", step)
        :return: updated pipeline estimator
        """
        return PipelineEstimator(steps=self._prepend_step(self.steps, step))

    def append_steps(self, to_add_steps: List[Step]) -> "PipelineEstimator":
        """
        Create pipeline estimator with steps list updated appending to add steps.

        :param to_add_steps: list of tuples of the form ("name", step)
        :return: updated pipeline estimator
        """
        return PipelineEstimator(
            steps=reduce(self._append_step, to_add_steps, self.steps)
        )

    def prepend_steps(self, to_add_steps: List[Step]) -> "PipelineEstimator":
        """
        Create pipeline estimator with steps list updated prepending to add steps.

        :param to_add_steps: list of tuples of the form ("name", step)
        :return: updated pipeline estimator
        """
        return PipelineEstimator(
            steps=reduce(self._prepend_step, reversed(to_add_steps), self.steps)
        )

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.

        :param deep: If True, return the parameters for this estimator and contained sub-objects that are estimators.
        :return: mapping of string to any parameter names mapped to their values.
        """
        return self._get_params("steps", deep=deep)

    def set_params(self, **kwargs) -> "PipelineEstimator":
        """
        Set the parameters of this estimator. Valid parameter keys can be listed with ``get_params()``.

        :param kwargs: additional keyworded parameters
        :return: self
        """
        self._set_params("steps", **kwargs)
        return self

    def __add__(self, other: "PipelineEstimator") -> "PipelineEstimator":
        """
        Concatenate pipelines.

        :param other: pipeline estimator

        :return: New concatenated PipelineEstimator
        """
        return self.append_steps(other.steps)

    def train(self, dataset: TDatasetUtilsMixin) -> "PipelineTransformer":
        """
        Train the estimator on the given dataset.

        :param dataset: input dataset. It contains both train features and train labels, already aligned.

        :return: Transformer obtained after training the estimator

        :raises TypeError: if step is not of Transformer type
        """
        tmp = dataset

        names, estimators = zip(*self.steps)

        transformers = estimators[:-1]
        last = estimators[-1]

        steps = []

        for stage, name in zip(transformers, names[:-1]):
            trans = stage.train(tmp) if isinstance(stage, Estimator) else stage

            if isinstance(stage, Resampler):
                tmp = trans.resample(tmp)
            else:
                steps.append((name, trans))
                tmp = trans.transform(tmp)

        steps.append(
            (names[-1], last.train(tmp) if isinstance(last, Estimator) else last)
        )

        if isinstance(steps[-1][1], Transformer):
            return PipelineTransformer(steps, estimator=self)
        else:
            raise TypeError("Last step of trained pipeline must be of type Transformer")
