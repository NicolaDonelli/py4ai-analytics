import os
import unittest

import numpy as np
import pandas as pd
from py4ai.data.model.ml import PandasDataset, PandasTimeIndexedDataset
from py4ai.core.tests.core import TestCase, logTest

from py4ai.analytics.ml.wrapper.sklearn.estimator import (
    KNNImputerEstimator,
    OneHotEncoderEstimator,
    Estimator as SklearnEstimator,
)

from py4ai.analytics.ml.wrapper.sklearn.wrapper import KNNImputer, OneHotEncoder

from py4ai.analytics.ml.wrapper.sklearn.estimator import MultiOutputEstimator
from py4ai.analytics.ml.wrapper.sklearn.splitter import SklearnSplitter
from py4ai.analytics.ml.wrapper.sklearn.wrapper import (
    KFold,
    MultiOutputClassifier,
    RandomForestClassifier,
)

from tests import DATA_FOLDER


class SklearnTest(TestCase):
    df: pd.DataFrame
    pdtidsd: PandasTimeIndexedDataset
    df_discr_label: pd.DataFrame
    ds_discr_label: PandasTimeIndexedDataset

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.read_csv(
            os.path.join(DATA_FOLDER, "weather_nyc_short.csv"), index_col="Date"
        )
        cls.pdtidsd = PandasTimeIndexedDataset(
            features=cls.df.drop("TempM", axis=1), labels=(cls.df.TempM > 50).map(int)
        )

        cls.df_discr_label = cls.df.copy()
        cls.df_discr_label.TempM = cls.df_discr_label.apply(
            lambda x: 0 if x.TempM < cls.df_discr_label.TempM.median() else 1, axis=1
        )
        cls.ds_discr_label = PandasTimeIndexedDataset(
            features=cls.df_discr_label.drop("TempM", axis=1),
            labels=cls.df_discr_label.TempM,
        )

    @logTest
    def test_predict_proba_SklearnEstimator(self) -> None:
        estimator = SklearnEstimator(
            RandomForestClassifier(random_state=42, n_estimators=100)
        )

        trained = estimator.train(self.ds_discr_label)

        predict_proba = trained.transform(self.ds_discr_label)

        self.assertGreater(len(predict_proba), 0)

    @logTest
    def test_KNNImputer(self) -> None:
        train_features = pd.DataFrame(
            {"c1": [1, 1, 2, 2, 4, 4, 2, 3], "c2": [2, 3, 4, 5, 6, 2, 1, 3]}
        )
        test_features = pd.DataFrame({"c1": [0, 1, 5, np.nan], "c2": [0, 3, np.nan, 3]})
        df_train_data = PandasDataset(train_features, None)
        df_test_data = PandasDataset(test_features, None)

        transformed = (
            KNNImputerEstimator(KNNImputer(n_neighbors=2), ["c2", "c1"])
            .train(df_train_data)
            .transform(df_test_data)
        )

        self.assertEqual(transformed.features.loc[2]["c2"], 4.0)

    @logTest
    def test_Dummy(self) -> None:
        train_features = pd.DataFrame(
            {
                "c1": [1, 1, 2, 2, 4, 4, 2, 3],
                "c2": ["a", "a", "b", "b", "b", "a", "b", "a"],
            }
        )
        test_features = pd.DataFrame({"c1": [0, 1, 5, 2], "c2": ["a", "b", "c", "a"]})
        df_train_data = PandasDataset(train_features, None)
        df_test_data = PandasDataset(test_features, None)

        oneHotEncoder = OneHotEncoderEstimator(
            OneHotEncoder(), min_freq={"c1": 0.2}, columns=["c1"]
        )
        transformer = oneHotEncoder.train(df_train_data)
        transformedTest = transformer.transform(df_test_data)

        self.assertListEqual(transformedTest.features["c1_1"].to_list(), [0, 1, 0, 0])
        self.assertListEqual(transformedTest.features["c1_2"].to_list(), [0, 0, 0, 1])
        self.assertListEqual(transformedTest.features["c1_4"].to_list(), [0, 0, 0, 0])
        self.assertListEqual(
            transformedTest.features["c2"].to_list(),
            df_test_data.features["c2"].to_list(),
        )

        oneHotEncoder = OneHotEncoderEstimator(
            OneHotEncoder(), min_freq={"c1": 1}, columns=["c1", "c2"]
        )
        transformer = oneHotEncoder.train(df_train_data)
        transformedTest = transformer.transform(df_test_data)

        self.assertListEqual(transformedTest.features["c1_1"].to_list(), [0, 1, 0, 0])
        self.assertListEqual(transformedTest.features["c1_2"].to_list(), [0, 0, 0, 1])
        self.assertListEqual(transformedTest.features["c1_4"].to_list(), [0, 0, 0, 0])
        self.assertListEqual(transformedTest.features["c2_a"].to_list(), [1, 0, 0, 1])
        self.assertListEqual(transformedTest.features["c2_b"].to_list(), [0, 1, 0, 0])

        train_features = np.transpose(
            np.array(
                [[1, 1, 2, 2, 4, 4, 2, 3], ["a", "a", "b", "b", "b", "a", "b", "a"]]
            )
        )
        test_features = np.transpose(np.array([[0, 1, 5, 2], ["a", "b", "c", "a"]]))
        transformedTest = (
            OneHotEncoder(handle_unknown="ignore")
            .train(train_features)
            .transform(test_features)
        )

        self.assertListEqual(transformedTest["0_1"].to_list(), [0, 1, 0, 0])
        self.assertListEqual(transformedTest["0_2"].to_list(), [0, 0, 0, 1])
        self.assertListEqual(transformedTest["1_a"].to_list(), [1, 0, 0, 1])
        self.assertListEqual(transformedTest["1_b"].to_list(), [0, 1, 0, 0])


class TestMultiOutputModels(TestCase):
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
    def test_multiOutputModels(self) -> None:

        multiLabels = pd.concat(
            [
                self.pdtidsd.labels["TempM"].to_frame("class_0"),
                1 - self.pdtidsd.labels["TempM"].to_frame("class_1"),
            ],
            axis=1,
        )

        multiClassDataset = PandasTimeIndexedDataset(self.pdtidsd.features, multiLabels)

        splitter = SklearnSplitter(KFold(n_splits=5))

        # train/test split
        train, test = next(splitter.split(multiClassDataset))

        multiClassEstimator = MultiOutputEstimator(
            MultiOutputClassifier(RandomForestClassifier())
        )

        model = multiClassEstimator.train(train)

        newDataset = model.transform(test)

        diff1 = np.abs(newDataset.labels - newDataset.features).sum().sum()
        diff2 = np.abs(test.labels - newDataset.features).sum().sum()

        self.assertEqual(diff1, diff2)

        # Prediction should not be perfect
        self.assertGreater(diff1, 0)

        unlabelledDataset = PandasTimeIndexedDataset(test.features, None)
        prediction = model.transform(unlabelledDataset)

        diff3 = np.abs(prediction.features - newDataset.features).sum().sum()

        self.assertEqual(diff3, 0)
        self.assertIsNotNone(newDataset.labels)
        self.assertIsNone(prediction.labels)

    @logTest
    def test_multiOutputModels_single_class(self) -> None:

        multiLabels = pd.concat(
            [self.pdtidsd.labels["TempM"].to_frame("class_0")],
            axis=1,
        )

        multiClassDataset = PandasTimeIndexedDataset(self.pdtidsd.features, multiLabels)

        splitter = SklearnSplitter(KFold(n_splits=5))

        # train/test split
        train, test = next(splitter.split(multiClassDataset))

        multiClassEstimator = MultiOutputEstimator(
            MultiOutputClassifier(RandomForestClassifier())
        )

        model = multiClassEstimator.train(train)

        newDataset = model.transform(test)

        diff1 = np.abs(newDataset.labels - newDataset.features).sum().sum()
        diff2 = np.abs(test.labels - newDataset.features).sum().sum()

        self.assertEqual(diff1, diff2)

        # Prediction should not be perfect
        self.assertGreater(diff1, 0)

        unlabelledDataset = PandasTimeIndexedDataset(test.features, None)
        prediction = model.transform(unlabelledDataset)

        diff3 = np.abs(prediction.features - newDataset.features).sum().sum()

        self.assertEqual(diff3, 0)
        self.assertIsNotNone(newDataset.labels)
        self.assertIsNone(prediction.labels)


if __name__ == "__main__":
    unittest.main()
