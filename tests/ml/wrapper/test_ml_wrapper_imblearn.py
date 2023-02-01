import os
import unittest

import pandas as pd
from py4ai.data.model.ml import PandasDataset

from py4ai.analytics.ml.wrapper.imblearn.wrapper import (
    SMOTE,
    RandomOverSampler,
    RandomUnderSampler,
)
from tests import DATA_FOLDER
from py4ai.core.tests.core import TestCase, logTest


class ImblearnWrappersTest(TestCase):
    df: pd.DataFrame
    exp_sampl_strat = 0.5

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.read_pickle(os.path.join(DATA_FOLDER, "imblearn_test.pkl"))

    @logTest
    def test_RandomOverSampler(self) -> None:
        df_resampled = RandomOverSampler(
            sampling_strategy=self.exp_sampl_strat
        ).resample(self.df)
        to_check = sum(df_resampled.labels.LABEL == 1) / sum(
            df_resampled.labels.LABEL == 0
        )

        self.assertIsInstance(df_resampled, PandasDataset)
        self.assertEqual(round(to_check, 1), self.exp_sampl_strat)

    @logTest
    def test_SMOTE(self) -> None:
        df_resampled = SMOTE(sampling_strategy=self.exp_sampl_strat).resample(self.df)
        to_check = sum(df_resampled.labels.LABEL == 1) / sum(
            df_resampled.labels.LABEL == 0
        )

        self.assertIsInstance(df_resampled, PandasDataset)
        self.assertEqual(round(to_check, 1), self.exp_sampl_strat)

    @logTest
    def test_RandomUnderSampler(self) -> None:
        df_resampled = RandomUnderSampler(
            sampling_strategy=self.exp_sampl_strat
        ).resample(self.df)
        to_check = sum(df_resampled.labels.LABEL == 1) / sum(
            df_resampled.labels.LABEL == 0
        )

        self.assertIsInstance(df_resampled, PandasDataset)
        self.assertEqual(round(to_check, 1), self.exp_sampl_strat)


if __name__ == "__main__":
    unittest.main()
