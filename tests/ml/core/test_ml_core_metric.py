import os
import unittest

import pandas as pd
from py4ai.data.model.ml import PandasTimeIndexedDataset
from py4ai.core.tests.core import TestCase, logTest

from py4ai.analytics.ml.core import Transformer
from py4ai.analytics.ml.core.metric import (
    LossMetric,
    Report,
    ScoreMetric,
    transformerScores,
)
from py4ai.analytics.ml.wrapper.sklearn.estimator import Estimator as SklearnEstimator
from py4ai.analytics.ml.wrapper.sklearn.evaluator import BinaryClassificationScorer
from py4ai.analytics.ml.wrapper.sklearn.wrapper import RandomForestClassifier

from tests import DATA_FOLDER


class TestBinaryClassifierScorer(TestCase):
    dataset: PandasTimeIndexedDataset
    estimator: SklearnEstimator
    transformer: Transformer
    evaluator: BinaryClassificationScorer

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(
            os.path.join(DATA_FOLDER, "weather_nyc_short.csv"), index_col="Date"
        )
        cls.dataset = PandasTimeIndexedDataset(
            features=df.drop("TempM", axis=1),
            labels=(df.TempM > 50).map(int).to_frame(1),
        )

        estimator = SklearnEstimator(skclass=RandomForestClassifier())
        cls.transformer = estimator.train(cls.dataset)
        cls.evaluator = BinaryClassificationScorer(bin_thresh=0.5, target_label=1)

    @logTest
    def test_transformerScores(self) -> None:
        scorer = transformerScores(self.transformer)

        score = scorer(self.evaluator.withMainMetric("auc"), self.dataset)

        self.assertIsInstance(score, Report)

    @logTest
    def test_score_creation_and_comparison(self) -> None:
        self.assertLess(ScoreMetric(0.4), ScoreMetric(0.5))

        self.assertGreater(ScoreMetric(0.6), ScoreMetric(0.5))

        self.assertLessEqual(ScoreMetric(0.5), ScoreMetric(0.5))

    @logTest
    def test_loss_creation_and_comparison(self) -> None:
        self.assertLess(LossMetric(0.6), LossMetric(0.5))

        self.assertGreater(LossMetric(0.4), LossMetric(0.5))

        self.assertGreaterEqual(LossMetric(0.5), LossMetric(0.5))


if __name__ == "__main__":
    unittest.main()
