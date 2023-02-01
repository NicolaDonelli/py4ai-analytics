import os
import unittest

import pandas as pd
from py4ai.data.model.ml import PandasTimeIndexedDataset, PandasDataset
from py4ai.core.tests.core import TestCase, logTest
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_tweedie_deviance,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from py4ai.analytics.ml.core.metric import LossMetric, Report, ScoreMetric
from py4ai.analytics.ml.wrapper.sklearn.estimator import GenericTransformer
from py4ai.analytics.ml.wrapper.sklearn.estimator import Estimator as SklearnEstimator
from py4ai.analytics.ml.wrapper.sklearn.evaluator import (
    BinaryClassificationScorer,
    RegressionScorer,
)
from py4ai.analytics.ml.wrapper.sklearn.wrapper import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from tests import DATA_FOLDER


class TestBinaryClassifierScorer(TestCase):
    df: pd.DataFrame
    dataset: PandasTimeIndexedDataset

    estimator: SklearnEstimator
    transformer: GenericTransformer

    evaluator = BinaryClassificationScorer(bin_thresh=0.5, target_label=1)

    predictions: PandasDataset
    predictedLabels: PandasDataset

    y_true: pd.Series
    y_pred: pd.DataFrame
    y_prob: pd.Series

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.read_csv(
            os.path.join(DATA_FOLDER, "weather_nyc_short.csv"), index_col="Date"
        )
        cls.dataset = PandasTimeIndexedDataset(
            features=cls.df.drop("TempM", axis=1),
            labels=(cls.df.TempM > 50).map(int).to_frame(1),
        )

        cls.estimator = SklearnEstimator(skclass=RandomForestClassifier())
        cls.transformer = cls.estimator.train(cls.dataset)

        cls.evaluator = BinaryClassificationScorer(bin_thresh=0.5, target_label=1)

        cls.predictions = cls.transformer.transform(cls.dataset)
        cls.predictedLabels = cls.evaluator._discretizer(0.5).transform(cls.predictions)

        cls.y_true = cls.predictions.getLabelsAs("pandas")
        cls.y_pred = cls.predictions.getFeaturesAs("pandas").idxmax(axis=1).to_frame(1)
        cls.y_prob = cls.predictions.getFeaturesAs("pandas")[1]

    @logTest
    def test_precision(self) -> None:
        self.assertEqual(
            self.evaluator.target_evaluators["precision"].evaluate(
                self.predictedLabels
            ),
            ScoreMetric(precision_score(y_true=self.y_true, y_pred=self.y_pred)),
        )

    @logTest
    def test_accuracy(self) -> None:
        self.assertEqual(
            self.evaluator.target_evaluators["accuracy"].evaluate(self.predictedLabels),
            ScoreMetric(accuracy_score(y_true=self.y_true, y_pred=self.y_pred)),
        )

    @logTest
    def test_recall(self) -> None:
        self.assertEqual(
            self.evaluator.target_evaluators["recall"].evaluate(self.predictedLabels),
            ScoreMetric(recall_score(y_true=self.y_true, y_pred=self.y_pred)),
        )

    @logTest
    def test_f1(self) -> None:
        self.assertEqual(
            self.evaluator.target_evaluators["f1"].evaluate(self.predictedLabels),
            ScoreMetric(f1_score(y_true=self.y_true, y_pred=self.y_pred)),
        )

    @logTest
    def test_auc(self) -> None:
        self.assertEqual(
            self.evaluator.score_evaluators["auc"].evaluate(self.predictions),
            ScoreMetric(roc_auc_score(y_true=self.y_true, y_score=self.y_prob)),
        )

    @logTest
    def test_metrics(self) -> None:
        to_check = {
            "accuracy": ScoreMetric(
                accuracy_score(y_true=self.y_true, y_pred=self.y_pred)
            ).value,
            "f1": ScoreMetric(f1_score(y_true=self.y_true, y_pred=self.y_pred)).value,
            "auc": ScoreMetric(
                roc_auc_score(y_true=self.y_true, y_score=self.y_pred)
            ).value,
            "precision": ScoreMetric(
                precision_score(y_true=self.y_true, y_pred=self.y_pred)
            ).value,
            "recall": ScoreMetric(
                recall_score(y_true=self.y_true, y_pred=self.y_pred)
            ).value,
        }

        report = self.evaluator.evaluate(self.predictions)

        self.assertIsInstance(report, Report)

        self.assertDictEqual(
            {k: v.value for k, v in sorted(report.metrics.items())},
            dict(sorted(to_check.items())),
        )


class TestRegressorScorer(TestCase):
    df: pd.DataFrame
    dataset: PandasTimeIndexedDataset

    estimator: SklearnEstimator
    transformer: GenericTransformer

    evaluator = BinaryClassificationScorer(bin_thresh=0.5, target_label=1)

    predictions: PandasDataset
    predictedLabels: PandasDataset

    y_true: pd.Series
    y_pred: pd.DataFrame
    y_prob: pd.Series

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.read_csv(
            os.path.join(DATA_FOLDER, "weather_nyc_short.csv"), index_col="Date"
        )
        cls.dataset = PandasTimeIndexedDataset(
            features=cls.df.drop("TempM", axis=1), labels=cls.df.TempM.to_frame("pred")
        )

        cls.estimator = SklearnEstimator(skclass=RandomForestRegressor())
        cls.transformer = cls.estimator.train(cls.dataset)

        cls.predictions = cls.transformer.transform(cls.dataset)

        cls.evaluator = RegressionScorer(target_label="pred")

        cls.y_true = cls.predictions.getLabelsAs("pandas")
        cls.y_pred = cls.predictions.getFeaturesAs("pandas")

    @logTest
    def test_RMSE_metric(self) -> None:
        self.assertEqual(
            self.evaluator.evaluators["root_mean_squared_error"].evaluate(
                self.predictions
            ),
            LossMetric(
                mean_squared_error(
                    y_true=self.y_true, y_pred=self.y_pred, squared=False
                )
            ),
        )

    @logTest
    def test_MAE_metric(self) -> None:
        self.assertEqual(
            self.evaluator.evaluators["mean_absolute_error"].evaluate(self.predictions),
            LossMetric(mean_absolute_error(y_true=self.y_true, y_pred=self.y_pred)),
        )

    @logTest
    def test_MAPE_metric(self) -> None:
        self.assertEqual(
            self.evaluator.evaluators["mean_absolute_percentage_error"].evaluate(
                self.predictions
            ),
            LossMetric(
                mean_absolute_percentage_error(y_true=self.y_true, y_pred=self.y_pred)
            ),
        )

    @logTest
    def test_median_absolute_error_metric(self) -> None:
        self.assertEqual(
            self.evaluator.evaluators["median_absolute_error"].evaluate(
                self.predictions
            ),
            LossMetric(median_absolute_error(y_true=self.y_true, y_pred=self.y_pred)),
        )

    @logTest
    def test_R2_metric(self) -> None:
        self.assertEqual(
            self.evaluator.evaluators["r2"].evaluate(self.predictions),
            ScoreMetric(r2_score(y_true=self.y_true, y_pred=self.y_pred)),
        )

    @logTest
    def test_explained_variance_metric(self) -> None:
        self.assertEqual(
            self.evaluator.evaluators["explained_variance"].evaluate(self.predictions),
            ScoreMetric(
                explained_variance_score(y_true=self.y_true, y_pred=self.y_pred)
            ),
        )

    @logTest
    def test_max_error_metric(self) -> None:
        self.assertEqual(
            self.evaluator.evaluators["max_error"].evaluate(self.predictions),
            LossMetric(max_error(y_true=self.y_true, y_pred=self.y_pred)),
        )

    @logTest
    def test_mean_tweedie_deviance_metric(self) -> None:
        self.assertEqual(
            self.evaluator.evaluators["mean_tweedie_deviance"].evaluate(
                self.predictions
            ),
            LossMetric(mean_tweedie_deviance(y_true=self.y_true, y_pred=self.y_pred)),
        )

    @logTest
    def test_metrics(self) -> None:
        to_check = {
            "root_mean_squared_error": LossMetric(
                mean_squared_error(
                    y_true=self.y_true, y_pred=self.y_pred, squared=False
                )
            ),
            "mean_absolute_error": LossMetric(
                mean_absolute_error(y_true=self.y_true, y_pred=self.y_pred)
            ),
            "mean_absolute_percentage_error": LossMetric(
                mean_absolute_percentage_error(y_true=self.y_true, y_pred=self.y_pred)
            ),
            "median_absolute_error": LossMetric(
                median_absolute_error(y_true=self.y_true, y_pred=self.y_pred)
            ),
            "r2": ScoreMetric(r2_score(y_true=self.y_true, y_pred=self.y_pred)),
            "explained_variance": ScoreMetric(
                explained_variance_score(y_true=self.y_true, y_pred=self.y_pred)
            ),
            "max_error": LossMetric(max_error(y_true=self.y_true, y_pred=self.y_pred)),
            "mean_tweedie_deviance": LossMetric(
                mean_tweedie_deviance(y_true=self.y_true, y_pred=self.y_pred)
            ),
        }

        report = self.evaluator.evaluate(self.predictions)

        self.assertIsInstance(report, Report)

        self.assertEqual(report.metrics, to_check)


if __name__ == "__main__":
    unittest.main()
