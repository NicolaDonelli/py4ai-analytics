import os
import unittest

import pandas as pd
from py4ai.data.model.core import Range
from py4ai.core.utils.fs import create_dir_if_not_exists

from py4ai.analytics.ml.core.optimizer.gridsearch import OptimizerGrid
from py4ai.analytics.ml.process.model import ModelLab
from py4ai.analytics.ml.wrapper.sklearn.estimator import Estimator as SklearnEstimator
from py4ai.analytics.ml.wrapper.sklearn.splitter import SklearnSplitter
from py4ai.analytics.ml.wrapper.sklearn.wrapper import GroupKFold, RandomForestRegressor
from tests import DATA_FOLDER, TMP_FOLDER
from py4ai.core.tests.core import TestCase, logTest
from tests.ml.process.helpers import (
    CustomReport,
    TimeEvolvingTest,
    FeatureProcessingExample,
    FeatureProcessingTimeSeriesExample,
)


class TestFeatureProcessing(TestCase):
    df: pd.DataFrame
    proc: FeatureProcessingExample

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.read_csv(
            os.path.join(DATA_FOLDER, "weather_nyc_short.csv"), index_col="Date"
        )
        cls.proc = FeatureProcessingExample(cls.df)

    @logTest
    def test_feature_processing_random_split(self) -> None:

        _, test = self.proc.random_split(test_fraction=0.2)
        self.assertEqual(test.features.shape[0], round(len(self.df.index) * 0.2))

        _, test = self.proc.random_split(test_fraction=0.1)
        self.assertEqual(test.features.shape[0], round(len(self.df.index) * 0.1))

        n = 500
        subrange = range(n)
        _, test = self.proc.random_split(test_fraction=0.2, subrange=subrange)
        self.assertEqual(test.features.shape[0], round(n * 0.2))

    @logTest
    def test_split_ranges(self) -> None:

        train_range = range(0, 800)
        test_range = range(801, len(self.df) - 1)

        train, test = self.proc.split_by_indices(train_range, test_range)

        self.assertEqual(set(train_range), set(train.features.index))
        self.assertEqual(set(test_range), set(test.features.index))


class TestModelLab(TestCase):
    df: pd.DataFrame
    model_path: str
    proc: FeatureProcessingTimeSeriesExample
    evaluator: CustomReport

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.read_csv(
            os.path.join(DATA_FOLDER, "weather_nyc_short.csv"), index_col="Date"
        )
        cls.proc = FeatureProcessingTimeSeriesExample(cls.df)
        cls.model_path = create_dir_if_not_exists(
            os.path.join(TMP_FOLDER, "models", "labs")
        )
        cls.evaluator = CustomReport(target_label="pred", main_metric="RMSE")

    @logTest
    def test_feature_processing_random_split(self) -> None:

        _, test = self.proc.random_split(test_fraction=0.2)
        self.assertEqual(test.features.shape[0], round(len(self.df.index) * 0.2))

        _, test = self.proc.random_split(test_fraction=0.1)
        self.assertEqual(test.features.shape[0], round(len(self.df.index) * 0.1))

        n = 500
        subrange = [pd.to_datetime(x) for x in self.df.index[:n]]
        _, test = self.proc.random_split(test_fraction=0.2, subrange=subrange)
        self.assertEqual(test.features.shape[0], round(n * 0.2))

    @logTest
    def test_split_by_range(self) -> None:

        train_range = Range(self.df.index[0], self.df.index[800])
        test_range = Range(self.df.index[801], self.df.index[-1])

        train, test = self.proc.split_by_time_range(train_range, test_range)

        self.assertEqual(set(train_range.days), set(train.features.index))
        self.assertEqual(set(test_range.days), set(test.features.index))

    @logTest
    def test_split_ranges(self) -> None:

        train_range = Range(self.df.index[0], self.df.index[800])
        test_range = Range(self.df.index[801], self.df.index[-1])

        train, test = self.proc.split_by_indices(train_range.days, test_range.days)

        self.assertEqual(set(train_range.days), set(train.features.index))
        self.assertEqual(set(test_range.days), set(test.features.index))

    @logTest
    def test_modelling_time_range(self) -> None:

        train_range = Range(self.df.index[0], self.df.index[800])
        test_range = Range(self.df.index[801], self.df.index[-1])

        mod = SklearnEstimator(skclass=RandomForestRegressor(random_state=42))
        split_strategy = SklearnSplitter(skclass=GroupKFold(n_splits=4))

        parameters_space = {
            "skclass__n_estimators": [10],
            "skclass__max_features": [1.0, 0.5],
            "skclass__max_depth": [3, 5],
        }

        opt = OptimizerGrid(
            estimator=mod,
            parameter_space=parameters_space,
            evaluator=self.evaluator,
            split_strategy=split_strategy,
            checkpoints=50,
        ).set_store_predictions(True)

        tester = TimeEvolvingTest(step=5, test_start=test_range.start)

        lab = (
            ModelLab(self.proc, opt, tester)
            .set_path(self.model_path)
            .set_name("my_run")
        )

        lab.execute(train_range, test_range)

    @logTest
    def test_modelling_indices(self) -> None:
        train_range = Range(self.df.index[0], self.df.index[800])
        test_range = Range(self.df.index[801], self.df.index[-1])

        mod = SklearnEstimator(skclass=RandomForestRegressor(random_state=42))
        split_strategy = SklearnSplitter(skclass=GroupKFold(n_splits=4))

        parameters_space = {
            "skclass__n_estimators": [10],
            "skclass__max_features": [1.0],
            "skclass__max_depth": [3],
        }

        opt = OptimizerGrid(
            estimator=mod,
            parameter_space=parameters_space,
            evaluator=self.evaluator,
            split_strategy=split_strategy,
            checkpoints=50,
        ).set_store_predictions(True)

        tester = TimeEvolvingTest(step=10, test_start=test_range.start)

        lab = (
            ModelLab(self.proc, opt, tester)
            .set_path(self.model_path)
            .set_name("my_run")
        )

        lab.execute(train_range.days, test_range.days)


if __name__ == "__main__":
    unittest.main()
