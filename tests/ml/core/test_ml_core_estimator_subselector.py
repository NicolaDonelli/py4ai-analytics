import os
import unittest
import pandas as pd
from py4ai.data.model.ml import PandasTimeIndexedDataset
from py4ai.core.tests.core import TestCase, logTest

from py4ai.analytics.ml.core.enricher.transformer.selector import (
    FeatureSelector,
    LabelSelector,
)
from py4ai.analytics.ml.wrapper.sklearn.estimator import Estimator as SklearnEstimator
from py4ai.analytics.ml.core.estimator.subselector import SubselectionEstimator
from py4ai.analytics.ml.wrapper.sklearn.wrapper import RandomForestClassifier
from py4ai.analytics.ml.core.pipeline.transformer import PipelineTransformer
from py4ai.analytics.ml.wrapper.sklearn.estimator import (
    GenericTransformer as SklearnTransformer,
)
from tests import DATA_FOLDER


class SubselectionEstimatorTest(TestCase):
    pdtidsd: PandasTimeIndexedDataset

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(
            os.path.join(DATA_FOLDER, "weather_nyc_short.csv"), index_col="Date"
        )
        cls.pdtidsd = PandasTimeIndexedDataset(
            features=df.drop("TempM", axis=1), labels=(df.TempM > 50).map(int)
        )

    @logTest
    def test_default_init(self) -> None:
        sse = SubselectionEstimator(
            estimator=SklearnEstimator(skclass=RandomForestClassifier())
        )
        self.assertEqual(len(sse.pipeline.steps), 1)
        self.assertEqual(sse.pipeline.steps[0][0], "est")
        self.assertEqual(sse.pipeline.steps[0][1], sse.estimator)

    @logTest
    def test_estimator_type(self) -> None:
        sse = SubselectionEstimator(
            features_name=["PressureA", "DewPointM"],
            labels_name=["TempM"],
            estimator=SklearnEstimator(skclass=RandomForestClassifier()),
        )
        self.assertIsInstance(sse.estimator, SklearnEstimator)

    @logTest
    def test_steps_names(self) -> None:
        sse = SubselectionEstimator(
            features_name=["PressureA", "DewPointM"],
            labels_name=["TempM"],
            estimator=SklearnEstimator(skclass=RandomForestClassifier()),
        )
        self.assertListEqual(
            [x[0] for x in sse.pipeline.steps],
            ["labelSelector", "featureSelector", "est"],
        )

    @logTest
    def test_steps_classes(self) -> None:
        sse = SubselectionEstimator(
            features_name=["PressureA", "DewPointM"],
            labels_name=["TempM"],
            estimator=SklearnEstimator(skclass=RandomForestClassifier()),
        )
        self.assertIsInstance(sse.pipeline.steps[0][1], LabelSelector)
        self.assertIsInstance(sse.pipeline.steps[1][1], FeatureSelector)
        self.assertEqual(sse.pipeline.steps[2][1], sse.estimator)

    @logTest
    def test_trained_class(self) -> None:
        sse = SubselectionEstimator(
            features_name=["PressureA", "DewPointM"],
            labels_name=["TempM"],
            estimator=SklearnEstimator(skclass=RandomForestClassifier()),
        )
        ssm = sse.train(self.pdtidsd)
        self.assertIsInstance(ssm, PipelineTransformer)

    @logTest
    def test_trained_step_classes(self) -> None:
        sse = SubselectionEstimator(
            features_name=["PressureA", "DewPointM"],
            labels_name=["TempM"],
            estimator=SklearnEstimator(skclass=RandomForestClassifier()),
        )
        ssm = sse.train(self.pdtidsd)

        self.assertIsInstance(ssm.steps[0][1], LabelSelector)
        self.assertIsInstance(ssm.steps[1][1], FeatureSelector)
        self.assertIsInstance(ssm.steps[2][1], SklearnTransformer)

    @logTest
    def test_wrong_features_name(self) -> None:
        sse = SubselectionEstimator(
            features_name=["Mario", "DewPointM"],
            labels_name=["TempM"],
            estimator=SklearnEstimator(skclass=RandomForestClassifier()),
        )
        self.assertRaisesRegex(KeyError, ".* not in index", sse.train, self.pdtidsd)


if __name__ == "__main__":
    unittest.main()
