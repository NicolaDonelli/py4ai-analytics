import os
import random
import unittest
from math import factorial

import numpy as np
import pandas as pd
from py4ai.data.model.ml import PandasDataset
from py4ai.core.tests.core import TestCase, logTest

from py4ai.analytics.ml.core.estimator.subselector import SubselectionEstimator
from py4ai.analytics.ml.core.optimizer.feature_search import OptimizerFeatureSearch
from py4ai.analytics.ml.core.optimizer.gridsearch import OptimizerGrid
from py4ai.analytics.ml.core.optimizer.random import OptimizerRandom
from py4ai.analytics.ml.core.optimizer.recursive import (
    OptimizerOutput,
    RecursiveOptimizer,
)
from py4ai.analytics.ml.core.optimizer.stepwise_forward import OptimizerStepwiseForward
from py4ai.analytics.ml.core.pipeline.transformer import PipelineEstimator
from py4ai.analytics.ml.core.splitter.time_evolving import TimeEvolvingSplitter
from py4ai.analytics.ml.wrapper.sklearn.estimator import Estimator as SklearnEstimator
from py4ai.analytics.ml.wrapper.sklearn.splitter import SklearnSplitter
from py4ai.analytics.ml.wrapper.sklearn.wrapper import (
    PCA,
    GroupKFold,
    RandomForestRegressor,
)
from tests import DATA_FOLDER, TMP_FOLDER
from tests.helpers import CustomReport


class OptimizerTest(TestCase):
    est: SklearnEstimator
    evaluator: CustomReport
    split_strategy: SklearnSplitter
    checkpoints_path: str
    train: PandasDataset
    time_evolving_splitter: TimeEvolvingSplitter

    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoints_path = os.path.join(TMP_FOLDER, "checkpoints")

        df = pd.read_csv(
            os.path.join(DATA_FOLDER, "weather_nyc_short.csv"),
            index_col="Date",
            parse_dates=True,
        )
        train_len = 800

        cls.train = PandasDataset(
            features=df.drop("TempM", axis=1).iloc[:train_len],
            labels=df["TempM"].iloc[:train_len].to_frame("pred"),
        )

        cls.est = SklearnEstimator(skclass=RandomForestRegressor(random_state=42))
        cls.split_strategy = SklearnSplitter(skclass=GroupKFold(n_splits=4))
        cls.time_evolving_splitter = TimeEvolvingSplitter()
        cls.evaluator = CustomReport(target_label="pred", main_metric="RMSE")

    @logTest
    def test_OptimizerGrid_grid(self) -> None:
        parameters_space = {
            "skclass__n_estimators": [10],
            "skclass__max_features": [1.0, 0.5],
            "skclass__max_depth": [3, 5],
        }

        opt_grid = OptimizerGrid(
            estimator=self.est,
            parameter_space=parameters_space,
            evaluator=self.evaluator,
            split_strategy=self.split_strategy,
            checkpoints_path=self.checkpoints_path,
        )
        res = opt_grid.optimize(self.train)
        self.assertEqual(len(res.history.keys()), 4)

        # Try IO of OptimizerOutput

        optPath = os.path.join(TMP_FOLDER, "optimizer-output")

        myModel = res.write(optPath)

        new = OptimizerOutput.read(myModel)

        def toDataFrame(dataset: PandasDataset) -> pd.DataFrame:
            return pd.concat([dataset.features, dataset.labels], axis=1)

        trainingPreds = toDataFrame(new.trainingPredictions().intersection())

        diff = toDataFrame(res.trainingPredictions()) == trainingPreds

        self.assertTrue(diff.values.all())

    @logTest
    def test_OptimizerGrid_grid_complex(self) -> None:
        parameters_space = {
            "skclass__n_estimators": [10],
            "skclass__max_features": np.linspace(start=0.5, stop=1, num=5, dtype=float),
            "skclass__max_depth": np.linspace(start=3, stop=5, num=2, dtype=int),
        }

        opt_grid = OptimizerGrid(
            estimator=self.est,
            parameter_space=parameters_space,
            evaluator=self.evaluator,
            split_strategy=self.split_strategy,
            checkpoints_path=self.checkpoints_path,
        )
        res = opt_grid.optimize(self.train)

        self.assertEqual(len(res.history.keys()), 10)

    @logTest
    def test_OptimizerRandom_base(self) -> None:
        parameters_space = {
            "skclass__n_estimators": {
                "sampler": np.random.choice,
                "params": {"a": [5, 10]},
            },
            "skclass__max_features": {
                "sampler": np.random.uniform,
                "params": {"low": 0.0, "high": 1.0},
            },
            "skclass__max_depth": {
                "sampler": np.random.randint,
                "params": {"low": 3, "high": 6},
            },
        }

        opt = OptimizerRandom(
            estimator=self.est,
            parameter_space=parameters_space,
            evaluator=self.evaluator,
            split_strategy=self.split_strategy,
            checkpoints_path=self.checkpoints_path,
            max_iterations=10,
        )
        res = opt.optimize(self.train)

        self.assertLessEqual(len(res.history.keys()), 10)

    @logTest
    def test_OptimizerRandom_sampling_func(self) -> None:
        parameters_space = {
            "skclass__n_estimators": {
                "sampler": random.choice,
                "params": {"seq": [5, 10]},
            },
            "skclass__max_features": {
                "sampler": np.random.uniform,
                "params": {"size": 1, "low": 0.5, "high": 1.0},
            },
            "skclass__max_depth": {
                "sampler": np.random.randint,
                "params": {"low": 3, "high": 5, "size": 1},
                "type": "int",
            },
        }

        opt = OptimizerRandom(
            estimator=self.est,
            parameter_space=parameters_space,
            evaluator=self.evaluator,
            split_strategy=self.split_strategy,
            checkpoints_path=self.checkpoints_path,
            max_iterations=10,
        )
        res = opt.optimize(self.train)

        self.assertLessEqual(len(res.history.keys()), 10)

    @logTest
    def test_OptimizerRandom_replicability(self) -> None:
        parameters_space = {
            "skclass__n_estimators": {
                "sampler": np.random.choice,
                "params": {"a": [5, 10]},
            },
            "skclass__max_features": {
                "sampler": np.random.uniform,
                "params": {"size": 1, "low": 0.5, "high": 1.0},
            },
            "skclass__max_depth": {
                "sampler": np.random.randint,
                "params": {"low": 3, "high": 5, "size": 1},
            },
        }

        opt = OptimizerRandom(
            estimator=self.est,
            parameter_space=parameters_space,
            evaluator=self.evaluator,
            split_strategy=self.split_strategy,
            checkpoints_path=self.checkpoints_path,
            max_iterations=10,
        )
        random.seed(42)
        np.random.seed(42)
        res1 = opt.optimize(self.train)
        best1 = res1.history[opt._get_best()]["params"]

        np.random.seed(42)
        res2 = opt.optimize(self.train)
        best2 = res2.history[opt._get_best()]["params"]

        self.assertTrue(all([best2[k] == best1[k] for k in best1.keys()]))

    @logTest
    def test_RecursiveOptimizer_computation(self) -> None:
        parameters_space = {
            "pca__skclass__n_components": [1, 2, 3],
            "model__skclass__n_estimators": [10],
            "model__skclass__max_features": [1.0, 0.5],
            "model__skclass__max_depth": [3, 5, 7],
        }

        pipeline = PipelineEstimator(
            [
                ("pca", SklearnEstimator(skclass=PCA())),
                (
                    "model",
                    SklearnEstimator(skclass=RandomForestRegressor(random_state=42)),
                ),
            ]
        )

        opt_grid = RecursiveOptimizer(
            estimator=pipeline,
            parameter_space=parameters_space,
            evaluator=self.evaluator,
            split_strategy=self.split_strategy,
            checkpoints_path=self.checkpoints_path,
        )

        res = opt_grid.optimize(self.train)

        self.assertEqual(len(res.history.keys()), 2 * 3 * 3)


class OptimizerFeatureSearchTest(TestCase):
    est: SklearnEstimator
    evaluator: CustomReport
    split_strategy: SklearnSplitter
    checkpoints_path: str
    train: PandasDataset
    time_evolving_splitter: TimeEvolvingSplitter
    base_opt: OptimizerFeatureSearch

    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoints_path = os.path.join(TMP_FOLDER, "checkpoints")

        df = pd.read_csv(
            os.path.join(DATA_FOLDER, "weather_nyc_short.csv"),
            index_col="Date",
            parse_dates=True,
        )
        train_len = 800

        cls.train = PandasDataset(
            features=df.drop("TempM", axis=1).iloc[:train_len],
            labels=df["TempM"].iloc[:train_len].to_frame("pred"),
        )

        cls.est = SklearnEstimator(skclass=RandomForestRegressor(random_state=42))
        cls.split_strategy = SklearnSplitter(skclass=GroupKFold(n_splits=4))
        cls.time_evolving_splitter = TimeEvolvingSplitter()
        cls.evaluator = CustomReport(target_label="pred", main_metric="RMSE")
        acc_evaluator = CustomReport(target_label="pred", main_metric="accuracy")
        cls.base_opt = OptimizerFeatureSearch(
            estimator=cls.est,
            evaluator=acc_evaluator,
            split_strategy=cls.split_strategy,
        ).set_store_predictions(True)

    @staticmethod
    def nCr(n, r) -> int:
        return int(factorial(n) / factorial(r) / factorial(n - r))

    @logTest
    def test_estimator_class(self) -> None:
        self.assertIsInstance(self.base_opt._estimator, SubselectionEstimator)

    @logTest
    def test_default_split_strategy(self) -> None:
        opt = OptimizerFeatureSearch(estimator=self.est, evaluator=self.evaluator)
        self.assertIsInstance(opt._split_strategy, TimeEvolvingSplitter)

    @logTest
    def test_tot_combinations(self) -> None:
        self.assertEqual(
            len(
                self.base_opt._generate_parameter_grid(
                    self.train.features.columns
                ).param_grid[0][self.base_opt._par_name]
            ),
            sum(
                self.nCr(self.train.features.shape[1], i)
                for i in range(1, self.train.features.shape[1] + 1)
            ),
        )

    @logTest
    def test_max_combinations(self) -> None:
        opt = OptimizerFeatureSearch(
            estimator=self.est,
            evaluator=self.evaluator,
            split_strategy=self.time_evolving_splitter,
            max_combinations=10,
        )
        self.assertEqual(
            len(
                opt._generate_parameter_grid(self.train.features.columns).param_grid[0][
                    self.base_opt._par_name
                ]
            ),
            10,
        )

    @logTest
    def test_max_features(self) -> None:
        max_features = 4
        opt = OptimizerFeatureSearch(
            estimator=self.est,
            evaluator=self.evaluator,
            split_strategy=self.time_evolving_splitter,
            max_features=max_features,
        )
        self.assertEqual(
            len(
                opt._generate_parameter_grid(self.train.features.columns).param_grid[0][
                    self.base_opt._par_name
                ]
            ),
            sum(
                self.nCr(self.train.features.shape[1], i)
                for i in range(1, max_features + 1)
            ),
        )

    @logTest
    def test_min_features(self) -> None:
        opt = OptimizerFeatureSearch(
            estimator=self.est,
            evaluator=self.evaluator,
            split_strategy=self.time_evolving_splitter,
            min_features=4,
        )
        self.assertEqual(
            len(
                opt._generate_parameter_grid(self.train.features.columns).param_grid[0][
                    self.base_opt._par_name
                ]
            ),
            sum(
                self.nCr(self.train.features.shape[1], i)
                for i in range(opt.min_features, self.train.features.shape[1] + 1)
            ),
        )

    @logTest
    def test_run(self) -> None:
        opt = OptimizerFeatureSearch(
            estimator=self.est,
            evaluator=self.evaluator,
            split_strategy=self.split_strategy,
            max_combinations=None,
        ).set_store_predictions(True)
        _ = opt.optimize(self.train)

        self.assertTrue(
            all(
                x in ["DewPointM", "Humidity"] for x in opt.best_estimator.features_name
            )
        )

    @logTest
    def test_reduced_run(self) -> None:
        cols = ["Humidity", "WindSpeedM", "VisibilityM"]
        opt = OptimizerFeatureSearch(
            estimator=self.est,
            evaluator=self.evaluator.withMainMetric("MAE"),
            split_strategy=self.split_strategy,
            features_subselection=cols,
        ).set_store_predictions(True)
        output = opt.optimize(self.train)
        self.assertTrue(all(output.dataset.features.columns == cols))

    @logTest
    def test_reduced_run_results(self) -> None:
        cols = ["Humidity", "WindSpeedM", "VisibilityM"]
        opt = OptimizerFeatureSearch(
            estimator=self.est,
            evaluator=self.evaluator.withMainMetric("MAE"),
            split_strategy=self.split_strategy,
            features_subselection=cols,
        ).set_store_predictions(True)
        _ = opt.optimize(self.train)
        self.assertEqual(
            set(["Humidity", "WindSpeedM", "VisibilityM"]),
            set(opt.best_estimator.features_name),
        )


class OptimizerStepwiseForwardTest(TestCase):
    est: SklearnEstimator
    evaluator: CustomReport
    split_strategy: SklearnSplitter
    checkpoints_path: str
    train: PandasDataset
    time_evolving_splitter: TimeEvolvingSplitter
    base_opt: OptimizerFeatureSearch

    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoints_path = os.path.join(TMP_FOLDER, "checkpoints")

        df = pd.read_csv(
            os.path.join(DATA_FOLDER, "weather_nyc_short.csv"),
            index_col="Date",
            parse_dates=True,
        )
        train_len = 800

        cls.train = PandasDataset(
            features=df.drop("TempM", axis=1).iloc[:train_len],
            labels=df["TempM"].iloc[:train_len].to_frame("pred"),
        )

        cls.est = SklearnEstimator(skclass=RandomForestRegressor(random_state=42))
        cls.split_strategy = SklearnSplitter(skclass=GroupKFold(n_splits=4))
        cls.time_evolving_splitter = TimeEvolvingSplitter()
        cls.evaluator = CustomReport(target_label="pred", main_metric="RMSE")
        acc_evaluator = CustomReport(target_label="pred", main_metric="accuracy")
        cls.base_opt = OptimizerStepwiseForward(
            estimator=cls.est,
            evaluator=acc_evaluator,
            split_strategy=cls.split_strategy,
        ).set_store_predictions(True)

    @staticmethod
    def nCr(n, r) -> int:
        return int(factorial(n) / factorial(r) / factorial(n - r))

    @logTest
    def test_estimator_class(self) -> None:
        self.assertIsInstance(self.base_opt._estimator, SubselectionEstimator)

    @logTest
    def test_default_split_strategy(self) -> None:
        opt = OptimizerStepwiseForward(
            estimator=self.est,
            evaluator=self.evaluator,
            split_strategy=self.split_strategy,
        )
        self.assertIsInstance(opt._split_strategy, SklearnSplitter)

    @logTest
    def test_init_combinations(self) -> None:
        self.assertEqual(
            len(
                self.base_opt._initial_vars(self.train.features.columns).param_grid[0][
                    self.base_opt._par_name
                ]
            ),
            self.train.features.shape[1],
        )

    @logTest
    def test_max_features(self) -> None:
        opt = OptimizerStepwiseForward(
            estimator=self.est,
            evaluator=self.evaluator,
            split_strategy=self.split_strategy,
            max_features=4,
        )
        opt.optimize(self.train)
        self.assertLessEqual(
            max(
                [
                    len(v["params"][self.base_opt._par_name])
                    for v in opt._history.values()
                ]
            ),
            opt.max_features,
        )

    @logTest
    def test_min_features(self) -> None:
        opt = OptimizerStepwiseForward(
            estimator=self.est,
            evaluator=self.evaluator,
            split_strategy=self.split_strategy,
            min_features=4,
        )
        self.assertEqual(
            len(
                opt._initial_vars(self.train.features.columns).param_grid[0][
                    self.base_opt._par_name
                ]
            ),
            self.nCr(self.train.features.shape[1], opt.min_features),
        )

    @logTest
    def test_run(self) -> None:
        opt = OptimizerStepwiseForward(
            estimator=self.est,
            evaluator=self.evaluator,
            split_strategy=self.split_strategy,
        ).set_store_predictions(True)
        _ = opt.optimize(self.train)

        self.assertTrue(
            all(
                x in ["DewPointM", "Humidity"] for x in opt.best_estimator.features_name
            )
        )

    @logTest
    def test_reduced_run(self) -> None:
        cols = ["Humidity", "WindSpeedM", "VisibilityM"]
        opt = OptimizerStepwiseForward(
            estimator=self.est,
            evaluator=self.evaluator.withMainMetric("MAE"),
            split_strategy=self.split_strategy,
            features_subselection=cols,
        ).set_store_predictions(True)
        output = opt.optimize(self.train)
        self.assertTrue(all(cols == output.dataset.features.columns))

    @logTest
    def test_reduced_run_result(self) -> None:
        cols = ["Humidity", "WindSpeedM", "VisibilityM"]
        opt = OptimizerStepwiseForward(
            estimator=self.est,
            evaluator=self.evaluator.withMainMetric("MAE"),
            split_strategy=self.split_strategy,
            features_subselection=cols,
        ).set_store_predictions(True)
        _ = opt.optimize(self.train)
        self.assertEqual(
            set(["Humidity", "VisibilityM", "WindSpeedM"]),
            set(opt.best_estimator.features_name),
        )


if __name__ == "__main__":
    unittest.main()
