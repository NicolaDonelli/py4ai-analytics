"""Implementation of metric classes."""
from dataclasses import dataclass
from typing import Callable, Dict, Union

from py4ai.data.model.ml import TDatasetUtilsMixin

from py4ai.analytics.ml.core import Evaluator, Metric, Transformer


@dataclass
class ScoreMetric(Metric):
    """
    Metric class to be used for scores, i.e. metrics that increase as predictive power increases.

    To be used when optimization shall maximize this metric.

    Attributes:
        value (float): value of the metric
    """

    score: float

    @property
    def value(self) -> float:
        """
        Score value.

        :return: score
        """
        return self.score


@dataclass
class LossMetric(Metric):
    """
    Metric class to be used for losses, i.e. metrics that decrease as predictive power increases.

    To be used when optimization shall minimize this metric.

    Attributes:
        value (float): value of the metric
    """

    loss: float

    @property
    def value(self):
        """
        Score value.

        :return: score
        """
        return -self.loss


class Report(Metric):
    """Class for wrapping multiple metrics into a single Metric class."""

    def __init__(
        self, metrics: Dict[str, Union[ScoreMetric, LossMetric]], main_metric: str
    ):
        """
        Class instance initializer.

        :param metrics: dictionary with the name of the metric and the Metric class (either a Loss or a Score)
        :param main_metric: name of the main metric to be used when optimizing performances
        :raises ValueError: provided metric unknown
        """
        if main_metric not in metrics:
            raise ValueError(
                "main_metric argument should be included in the list of metrics provided"
            )

        self.metrics = metrics
        self.main_metric = main_metric

    @property
    def value(self) -> float:
        """
        Score value.

        :return: score
        """
        return self.metrics[self.main_metric].value


def transformerScores(
    transformer: Transformer,
) -> Callable[[Evaluator, TDatasetUtilsMixin], Metric]:
    """
    Create a function to evaluate metrics for a given transformer.

    :param transformer: Provided transformer which is to be evaluated
    :return: evaluator function
    """

    def _evaluator(evaluator: Evaluator, dataset: TDatasetUtilsMixin) -> Metric:
        return evaluator.evaluate(transformer.transform(dataset))

    return _evaluator
