import os
import unittest

import pandas as pd
from py4ai.data.model.ml import PandasTimeIndexedDataset
from py4ai.core.tests.core import TestCase, logTest

from py4ai.analytics.ml.core.enricher.transformer.selector import FeatureSelector
from py4ai.analytics.ml.core.estimator.selector import DeleteDuplicates
from tests import DATA_FOLDER


class TestDeleteDuplicates(TestCase):
    pdtids: PandasTimeIndexedDataset

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(
            os.path.join(DATA_FOLDER, "weather_nyc_short.csv"), index_col="Date"
        )
        cls.pdtids = PandasTimeIndexedDataset(
            features=df.drop("TempM", axis=1), labels=df.TempM
        )

    @logTest
    def test_least_correlated_components(self) -> None:
        dd = DeleteDuplicates()
        tmp = dd._least_correlated_components(
            self.pdtids.features.corr(), ["DewPointM", "PressureA", "WindSpeedM"]
        )
        to_check = (
            self.pdtids.features[["WindSpeedM", "PressureA"]].corr().abs().iloc[0, 1],
            3,
        )
        self.assertEqual(to_check, tmp)

    @logTest
    def test_most_correlated(self) -> None:
        dd = DeleteDuplicates()
        tmp = dd._most_correlated(self.pdtids.features, self.pdtids.labels)
        to_check = (
            self.pdtids.features.corrwith(self.pdtids.labels.iloc[:, 0]).abs().idxmax(),
            self.pdtids.features.corrwith(self.pdtids.labels.iloc[:, 0]).abs().max(),
        )
        self.assertEqual(to_check, tmp)

    @logTest
    def test_graph_correlation(self) -> None:
        dd = DeleteDuplicates(threshold=0.5, min_threshold=0.2, step=0.05)
        tmp = sorted(dd._graph_correlation(self.pdtids).features.columns)
        to_check = ["DewPointM", "PressureA", "WindSpeedM"]
        self.assertEqual(to_check, tmp)

    @logTest
    def test_train(self) -> None:
        dd = DeleteDuplicates(threshold=0.5, min_threshold=0.2, step=0.05)
        selector = dd.train(self.pdtids)
        outdataset = selector.transform(self.pdtids)
        self.assertIsInstance(selector, FeatureSelector)
        self.assertEqual(
            outdataset.getFeaturesAs("pandas").sort_index(axis=1),
            self.pdtids.features[["DewPointM", "PressureA", "WindSpeedM"]],
        )
        self.assertEqual(outdataset.getLabelsAs("pandas"), self.pdtids.labels)

    @logTest
    def test_train_with_typing_error(self) -> None:
        dd = DeleteDuplicates(threshold=0.5, min_threshold=0.2, step=0.05)
        self.assertRaises(TypeError, dd.train, dataset=[])


if __name__ == "__main__":
    unittest.main()
