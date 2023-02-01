import os
import unittest

import pandas as pd
from py4ai.data.model.ml import PandasTimeIndexedDataset
from py4ai.core.tests.core import TestCase, logTest

from py4ai.analytics.ml.core.estimator.stacking import StackingEstimator
from py4ai.analytics.ml.core.splitter.time_evolving import TimeEvolvingSplitter
from py4ai.analytics.ml.process.resumer import TopModelsFromRunner
from py4ai.analytics.ml.process.runner import RunnerResults
from py4ai.analytics.ml.wrapper.sklearn.estimator import Estimator as SklearnEstimator
from py4ai.analytics.ml.wrapper.sklearn.splitter import SklearnSplitter
from py4ai.analytics.ml.wrapper.sklearn.wrapper import (
    GroupKFold,
    RandomForestRegressor,
    RandomForestClassifier,
)
from tests.helpers import MaxCombiner
from tests import DATA_FOLDER


class TestStackingEstimator(TestCase):
    pdtids: PandasTimeIndexedDataset
    pdtidsd: PandasTimeIndexedDataset
    discr_label: PandasTimeIndexedDataset

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(
            os.path.join(DATA_FOLDER, "weather_nyc_short.csv"), index_col="Date"
        )
        cls.pdtids = PandasTimeIndexedDataset(
            features=df.drop("TempM", axis=1), labels=df.TempM
        )
        cls.pdtidsd = PandasTimeIndexedDataset(
            features=df.drop("TempM", axis=1), labels=(df.TempM > 50).map(int)
        )

        df_discr_label = df.copy()
        df_discr_label.TempM = df_discr_label.apply(
            lambda x: 0 if x.TempM < df.TempM.median() else 1, axis=1
        )
        cls.discr_label = PandasTimeIndexedDataset(
            features=df_discr_label.drop("TempM", axis=1),
            labels=df_discr_label.TempM,
        )

    @logTest
    def test_ensembling(self) -> None:

        gkf = GroupKFold(
            partition_key=lambda x: x.date(),
            n_splits=3,
            shuffle=False,
            random_state=None,
        )
        split_strategy = SklearnSplitter(skclass=gkf)

        rfs = [
            ("model_1", RandomForestRegressor(random_state=42, n_estimators=100)),
            ("model_2", RandomForestRegressor(random_state=42, n_estimators=50)),
        ]

        ensembler = SklearnEstimator(RandomForestRegressor())

        models = StackingEstimator(
            [(name, SklearnEstimator(model)) for name, model in rfs],
            ensembler=ensembler,
            folding=split_strategy,
        )

        model = models.train(self.pdtids)

        predict = model.transform(self.pdtidsd).getFeaturesAs("pandas")

        self.assertEqual(predict.shape, self.pdtids.getLabelsAs("pandas").shape)

    @logTest
    def test_from_modeling(self) -> None:

        gkf = GroupKFold(
            partition_key=lambda x: x.date(),
            n_splits=3,
            shuffle=False,
            random_state=None,
        )
        split_strategy = SklearnSplitter(skclass=gkf)

        rfs = [
            ("model_1", RandomForestRegressor(random_state=42, n_estimators=100)),
            ("model_2", RandomForestRegressor(random_state=42, n_estimators=50)),
        ]

        ensembler = SklearnEstimator(RandomForestRegressor())

        models = StackingEstimator(
            [
                (name, SklearnEstimator(model).train(dataset=self.pdtids))
                for name, model in rfs
            ],
            ensembler=ensembler,
            folding=split_strategy,
        )

        model = models.train(self.pdtids)

        predict = model.transform(self.pdtidsd).getFeaturesAs("pandas")

        self.assertEqual(predict.shape, self.pdtids.getLabelsAs("pandas").shape)

    @logTest
    def test_from_modeling_mixed(self) -> None:
        gkf = GroupKFold(
            partition_key=lambda x: x.date(),
            n_splits=3,
            shuffle=False,
            random_state=None,
        )
        split_strategy = SklearnSplitter(skclass=gkf)

        rfs = [
            ("model_1", RandomForestRegressor(random_state=42, n_estimators=100)),
            ("model_2", RandomForestRegressor(random_state=42, n_estimators=50)),
        ]

        ensembler = SklearnEstimator(RandomForestRegressor())

        models = StackingEstimator(
            [
                (name, SklearnEstimator(model).train(dataset=self.pdtids))
                for name, model in rfs
            ]
            + [(name, SklearnEstimator(model)) for name, model in rfs],
            ensembler=ensembler,
            folding=split_strategy,
        )

        model = models.train(self.pdtids)

        predict = model.transform(self.pdtidsd).getFeaturesAs("pandas")

        self.assertEqual(predict.shape, self.pdtids.getLabelsAs("pandas").shape)

    @logTest
    def test_ensembler_transformer(self) -> None:

        gkf = GroupKFold(
            partition_key=lambda x: x.date(),
            n_splits=3,
            shuffle=False,
            random_state=None,
        )
        split_strategy = SklearnSplitter(skclass=gkf)

        rfs = [
            ("model_1", RandomForestRegressor(random_state=42, n_estimators=100)),
            ("model_2", RandomForestRegressor(random_state=42, n_estimators=50)),
        ]

        ensembler = MaxCombiner()

        models = StackingEstimator(
            [
                (name, SklearnEstimator(model).train(dataset=self.pdtids))
                for name, model in rfs
            ]
            + [(name, SklearnEstimator(model)) for name, model in rfs],
            ensembler=ensembler,
            folding=split_strategy,
        )

        model = models.train(self.pdtids)

        predict = model.transform(self.pdtidsd).getFeaturesAs("pandas")

        self.assertEqual(predict.shape, self.pdtids.getLabelsAs("pandas").shape)

        predict_1 = (
            model.steps[0][1]
            .transformers[0][2]
            .transform(self.pdtidsd)
            .getFeaturesAs("pandas")
        )
        predict_2 = (
            model.steps[0][1]
            .transformers[1][2]
            .transform(self.pdtidsd)
            .getFeaturesAs("pandas")
        )
        self.assertTrue(((predict == predict_1) | (predict == predict_2)).all().all())

    @logTest
    def test_ensembler_no_train(self) -> None:
        rfs = [
            ("model_1", RandomForestRegressor(random_state=42, n_estimators=100)),
            ("model_2", RandomForestRegressor(random_state=42, n_estimators=50)),
        ]

        ensembler = MaxCombiner()

        model = StackingEstimator.compose(
            [
                (name, SklearnEstimator(model).train(dataset=self.pdtids))
                for name, model in rfs
            ]
            + [
                (name, SklearnEstimator(model).train(dataset=self.pdtids))
                for name, model in rfs
            ],
            ensembler=ensembler,
        )

        predict = model.transform(self.pdtidsd).getFeaturesAs("pandas")

        self.assertEqual(predict.shape, self.pdtids.getLabelsAs("pandas").shape)

        predict_1 = (
            model.steps[0][1]
            .transformers[0][2]
            .transform(self.pdtidsd)
            .getFeaturesAs("pandas")
        )
        predict_2 = (
            model.steps[0][1]
            .transformers[1][2]
            .transform(self.pdtidsd)
            .getFeaturesAs("pandas")
        )
        self.assertTrue(((predict == predict_1) | (predict == predict_2)).all().all())

    @logTest
    def test_time_evolving_fold(self) -> None:
        split_strategy = TimeEvolvingSplitter(
            n_folds=10,
            train_ratio=0.9,
            min_periods_per_fold=1,
            window=None,
            valid_start=None,
            g=lambda x: x,
        )

        rfs = [
            ("model_1", RandomForestRegressor(random_state=42, n_estimators=100)),
            ("model_2", RandomForestRegressor(random_state=42, n_estimators=50)),
        ]

        ensembler = SklearnEstimator(RandomForestRegressor())

        models = StackingEstimator(
            [
                (name, SklearnEstimator(model).train(dataset=self.pdtids))
                for name, model in rfs
            ]
            + [(name, SklearnEstimator(model)) for name, model in rfs],
            ensembler=ensembler,
            folding=split_strategy,
        )

        model = models.train(self.pdtids)

        predict = model.transform(self.pdtidsd).getFeaturesAs("pandas")

        self.assertEqual(predict.shape, self.pdtids.getLabelsAs("pandas").shape)

    @logTest
    def test_from_path(self) -> None:
        split_strategy = TimeEvolvingSplitter(
            n_folds=10,
            train_ratio=0.9,
            min_periods_per_fold=1,
            window=None,
            valid_start=None,
            g=lambda x: x,
        )

        rfs = [
            ("model_1", RandomForestRegressor(random_state=42, n_estimators=100)),
            ("model_2", RandomForestRegressor(random_state=42, n_estimators=50)),
        ]

        ensembler = SklearnEstimator(RandomForestRegressor())

        models = StackingEstimator(
            [
                (name, SklearnEstimator(model).train(dataset=self.pdtids))
                for name, model in rfs
            ]
            + [("ext", os.path.join(DATA_FOLDER, "models", "labs", "my_run"))],
            ensembler=ensembler,
            folding=split_strategy,
        )

        model = models.train(self.pdtids)

        predict = model.transform(self.pdtidsd).getFeaturesAs("pandas")

        self.assertEqual(predict.shape, self.pdtids.getLabelsAs("pandas").shape)

    @logTest
    def test_from_model_path(self) -> None:
        split_strategy = TimeEvolvingSplitter(
            n_folds=10,
            train_ratio=0.9,
            min_periods_per_fold=1,
            window=None,
            g=lambda x: x,
        )

        rfs = [
            ("model_1", RandomForestRegressor(random_state=42, n_estimators=100)),
            ("model_2", RandomForestRegressor(random_state=42, n_estimators=50)),
        ]

        ensembler = SklearnEstimator(RandomForestRegressor())

        models = StackingEstimator(
            [
                (name, SklearnEstimator(model).train(dataset=self.pdtids))
                for name, model in rfs
            ]
            + [
                (
                    "runner",
                    TopModelsFromRunner(
                        RunnerResults(
                            os.path.join(DATA_FOLDER, "models", "labs", "my_run")
                        ),
                        3,
                    ),
                )
            ],
            ensembler=ensembler,
            folding=split_strategy,
        )

        model = models.train(self.pdtids)

        predict = model.transform(self.pdtidsd).getFeaturesAs("pandas")

        self.assertEqual(predict.shape, self.pdtids.getLabelsAs("pandas").shape)

    @logTest
    def test_predict_proba_StackingEstimator(self) -> None:
        gkf = GroupKFold(
            partition_key=lambda x: x.date(),
            n_splits=3,
            shuffle=False,
            random_state=None,
        )
        split_strategy = SklearnSplitter(skclass=gkf)

        rfs = [
            ("rf1", RandomForestClassifier(random_state=42, n_estimators=100)),
            ("rf2", RandomForestClassifier(random_state=42, n_estimators=50)),
        ]

        ensembler = SklearnEstimator(RandomForestClassifier())

        models = StackingEstimator(
            [(name, SklearnEstimator(model)) for name, model in rfs],
            ensembler=ensembler,
            folding=split_strategy,
        )

        model = models.train(self.discr_label)

        predict_proba = model.transform(self.discr_label)

        self.assertGreater(len(predict_proba), 0)


if __name__ == "__main__":
    unittest.main()
