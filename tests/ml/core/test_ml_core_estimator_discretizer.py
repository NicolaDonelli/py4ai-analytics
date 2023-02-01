from typing import Dict, List, Union
import pandas as pd
from py4ai.analytics.ml.core.estimator.discretizer import QuantileEstimator
from py4ai.analytics.ml.core.enricher.transformer.discretizer import Discretizer
import unittest
from py4ai.data.model.ml import PandasDataset
from py4ai.core.tests.core import TestCase, logTest


class TestQuantileEstimator(TestCase):
    to_discretize_q: Dict[str, Dict[str, Dict[str, Union[List[float], List[int]]]]]
    df: PandasDataset
    quantiles: List[float]
    thresholds: Dict[str, Dict[str, Dict[str, List[float]]]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.to_discretize_q = {
            "features": {"f_1": {"q_list": [0.3, 0.7], "label_names": [0, 1, 2]}}
        }
        cls.df = PandasDataset(
            features=pd.DataFrame(columns=["f_1"], data=[1, 1, 2, 2, 2, 2, 2, 3, 3, 3]),
            labels=pd.Series([1] * 10),
        )

        cls.quantiles = [
            cls.df.features["f_1"].quantile(i)
            for i in cls.to_discretize_q["features"]["f_1"]["q_list"]
        ]
        cls.thresholds = {
            "features": {
                "f_1": {
                    "threshold": cls.quantiles,
                    "label_names": cls.to_discretize_q["features"]["f_1"]["label_names"],
                }
            },
            "labels": {}
        }
        cls.qe = QuantileEstimator(cls.to_discretize_q)

    @logTest
    def test_train(self) -> None:
        qe_trained = self.qe.train(self.df)

        self.assertIsInstance(qe_trained, Discretizer)
        self.assertEqual(qe_trained.to_discretize, self.thresholds)

    @logTest
    def test_compute_threshold(self) -> None:
        self.assertEqual(
            sorted(self.quantiles),
            self.qe.compute_threshold(
                self.df.features["f_1"],
                self.to_discretize_q["features"]["f_1"]["q_list"],
            ),
        )

    @logTest
    def test_transform(self) -> None:

        discr_trained = self.qe.train(self.df)
        new_df = discr_trained.transform(self.df)

        self.assertEqual(new_df.features.shape, self.df.features.shape)
        self.assertEqual(new_df.labels.shape, self.df.labels.shape)


if __name__ == "__main__":
    unittest.main()
