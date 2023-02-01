"""Contains Evaluators wrappers."""
from typing import Callable, Dict, Optional, Type, Union

from py4ai.core.utils.decorators import lazyproperty as lazy
from py4ai.core.utils.dict import union
from sklearn.metrics import accuracy_score, explained_variance_score, f1_score
from sklearn.metrics import max_error as max_error_sk
from sklearn.metrics import mean_absolute_error as mean_absolute_error_sk
from sklearn.metrics import (
    mean_absolute_percentage_error as mean_absolute_percentage_error_sk,
)
from sklearn.metrics import mean_squared_error as mean_squared_error_sk
from sklearn.metrics import mean_tweedie_deviance as mean_tweedie_deviance_sk
from sklearn.metrics import median_absolute_error as median_absolute_error_sk
from sklearn.metrics import precision_score, r2_score, recall_score, roc_auc_score

from py4ai.analytics.ml import ArrayLike
from py4ai.analytics.ml.core import TDatasetUtilsMixin, Evaluator, Metric, Transformer
from py4ai.analytics.ml.core.estimator.discretizer import Discretizer
from py4ai.analytics.ml.core.metric import LossMetric, Report, ScoreMetric

MetricType = Type[Metric]
ErrorFunctionType = Callable[[ArrayLike, ArrayLike], float]


class PandasEvaluator(Evaluator):
    """Provide a general interface for creating Evaluators based on a scoring function, like the one provided by sklearn."""

    def __init__(
        self,
        func: ErrorFunctionType,
        type: MetricType,
        target_name: Optional[str] = None,
    ):
        """
        Initialize the class.

        :param func: Callable that scores predictions. It should have a form like (y_true, y_pred) => score
        :param type: Type of metric, whether Score or Loss, to be instantiated from the score.
                     This influences whether the function should be maximized or minimized for best performances
        :param target_name: Name of the column/field where the predictions to be used are contained
        """
        self.func = func
        self.type = type
        self.target_name = target_name

    def evaluate(self, dataset: TDatasetUtilsMixin) -> Metric:
        """
        Evaluate the metric.

        :param dataset: dataset to be evaluated
        :return: metrics specified in type input
        """
        (true, pred) = (
            (dataset.getLabelsAs("pandas"), dataset.getFeaturesAs("pandas"))
            if self.target_name is None
            else (
                dataset.getLabelsAs("pandas")[self.target_name],
                dataset.getFeaturesAs("pandas")[self.target_name],
            )
        )

        return self.type(self.func(true, pred))

    def withTargetName(self, name: str) -> "PandasEvaluator":
        """
        Create a new evaluator that makes use of a specific column/field for evaluation.

        :param name: name of the field/column to be used
        :return: new instance of the PandasEvaluator
        """
        return PandasEvaluator(self.func, self.type, name)


r2 = PandasEvaluator(r2_score, ScoreMetric)
auc = PandasEvaluator(roc_auc_score, ScoreMetric)
precision = PandasEvaluator(precision_score, ScoreMetric)
recall = PandasEvaluator(recall_score, ScoreMetric)
f1 = PandasEvaluator(f1_score, ScoreMetric)
accuracy = PandasEvaluator(accuracy_score, ScoreMetric)

explained_variance = PandasEvaluator(explained_variance_score, ScoreMetric)
mean_absolute_error = PandasEvaluator(mean_absolute_error_sk, LossMetric)
root_mean_squared_error = PandasEvaluator(
    lambda y_true, y_pred: mean_squared_error_sk(
        y_true=y_true, y_pred=y_pred, squared=False
    ),
    LossMetric,
)
max_error = PandasEvaluator(max_error_sk, LossMetric)
mean_tweedie_deviance = PandasEvaluator(mean_tweedie_deviance_sk, LossMetric)
mean_absolute_percentage_error = PandasEvaluator(
    mean_absolute_percentage_error_sk, LossMetric
)
median_absolute_error = PandasEvaluator(median_absolute_error_sk, LossMetric)


class BinaryClassificationScorer(Evaluator):
    """Evaluate binary classifier performances."""

    base_target_evaluators: Dict[str, PandasEvaluator] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

    base_score_evaluators: Dict[str, PandasEvaluator] = {"auc": auc}

    def __init__(
        self,
        bin_thresh: float = 0.5,
        target_label: Union[int, float, str] = 1,
        main_metric: Optional[str] = None,
    ):
        """
        Initialize the class.

        :param bin_thresh: threshold to use to bin probabilities
        :param target_label: name of the label to bin
        :param main_metric: name of the main metric to be used when optimizing performances
        """
        self.bin_thresh = bin_thresh
        self.target_label = target_label
        self.main_metric = main_metric if main_metric else "f1"

    @lazy
    def score_evaluators(self):
        """
        Return the evaluators to be used with prediction probabilities.

        :return: dictionary with the name of the metric and the evaluator that uses the target label
        """
        return {
            name: self._setTargetName(evaluator)
            for name, evaluator in self.base_score_evaluators.items()
        }

    @lazy
    def target_evaluators(self):
        """
        Return the evaluators to be used with prediction classes.

        :return: dictionary with the name of the metric and the evaluator that uses the target label
        """
        return {
            name: self._setTargetName(evaluator)
            for name, evaluator in self.base_target_evaluators.items()
        }

    def _discretizer(self, value) -> Transformer:
        return Discretizer(
            {
                "features": {
                    self.target_label: {"threshold": [value], "label_names": [0, 1]}
                }
            }
        )

    def _setTargetName(self, evaluator: PandasEvaluator):
        return (
            evaluator.withTargetName(self.target_label)
            if self.target_label
            else evaluator
        )

    @lazy
    def _allKeys(self):
        return set(self.base_score_evaluators).union(self.base_target_evaluators)

    def withMainMetric(self, main_metric: str):
        """
        Create a new evaluator that selects a different main_metric to be optimized.

        :param main_metric: name of the metric to be used (among the availables ones)
        :return: new instance of the Evaluator
        :raises ValueError: if provided metric is not known
        """
        if main_metric not in self._allKeys:
            raise ValueError(
                f"main_metric should be one of these possible values: "
                f"{','.join(self._allKeys)}"
            )

        cls = type(self)
        return cls(self.bin_thresh, self.target_label, main_metric)

    def evaluate(self, dataset: TDatasetUtilsMixin) -> Metric:
        """
        Evaluate the metrics.

        :param dataset: dataset to be evaluated
        :return: metrics report
        """
        binned = self._discretizer(self.bin_thresh).transform(dataset)

        return Report(
            union(
                {
                    name: evaluator.evaluate(binned)
                    for name, evaluator in self.target_evaluators.items()
                },
                {
                    name: evaluator.evaluate(dataset)
                    for name, evaluator in self.score_evaluators.items()
                },
            ),
            self.main_metric,
        )


class RegressionScorer(Evaluator):
    """Evaluate regressors performances."""

    base_evaluators: Dict[str, PandasEvaluator] = {
        "r2": r2,
        "explained_variance": explained_variance,
        "mean_absolute_error": mean_absolute_error,
        "root_mean_squared_error": root_mean_squared_error,
        "max_error": max_error,
        "mean_tweedie_deviance": mean_tweedie_deviance,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
        "median_absolute_error": median_absolute_error,
    }

    def __init__(
        self,
        target_label: Union[int, float, str] = 1,
        main_metric: Optional[str] = None,
    ):
        """
        Initialize the class.

        :param target_label: name of the label to bin
        :param main_metric: name of the main metric to be used when optimizing performances
        """
        self.target_label = target_label
        self.main_metric = main_metric if main_metric else "r2"

    @lazy
    def evaluators(self):
        """
        Return the evaluators to be used with the prediction.

        :return: dictionary with the name of the metric and the evaluator that uses the target label
        """
        return {
            name: self._setTargetName(evaluator)
            for name, evaluator in self.base_evaluators.items()
        }

    def _setTargetName(self, evaluator: PandasEvaluator):
        return (
            evaluator.withTargetName(self.target_label)
            if self.target_label
            else evaluator
        )

    def withMainMetric(self, main_metric):
        """
        Create a new evaluator that selects a different main_metric to be optimized.

        :param main_metric: name of the metric to be used (among the availables ones)
        :return: new instance of the Evaluator
        :raises ValueError: if the provided metric is not known
        """
        if main_metric not in self.base_evaluators:
            raise ValueError(
                f"main_metric should be one of these possible values: "
                f"{','.join(self.base_evaluators.keys())}"
            )

        cls = type(self)
        return cls(self.target_label, main_metric)

    def evaluate(self, dataset: TDatasetUtilsMixin) -> Metric:
        """
        Evaluate the metrics report.

        :param dataset: dateset to evaluated
        :return: report
        """
        return Report(
            {
                name: evaluator.evaluate(dataset)
                for name, evaluator in self.evaluators.items()
            },
            self.main_metric,
        )
