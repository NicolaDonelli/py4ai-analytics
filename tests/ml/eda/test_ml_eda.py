import os
import unittest
from datetime import timedelta

import pandas as pd

from py4ai.analytics.ml.eda.general import (
    computeCorrelation,
    filterColumnsWith,
    flattenMultiIndex,
    generate_multi_index
)
from py4ai.analytics.ml.eda.time import lagSeries
from tests import DATA_FOLDER
from py4ai.core.tests.core import TestCase, logTest


class PandasTest(TestCase):
    df: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.read_csv(
            os.path.join(DATA_FOLDER, "weather_nyc_short.csv"), index_col="Date"
        )
        cls.sep_flatten = '-&&-'

    def test_filterColumns(self):
        self.assertEqual(len(filterColumnsWith(self.df, "Pressure").columns), 1)
        self.assertTrue(
            all(
                [
                    "M" in columnName
                    for columnName in filterColumnsWith(self.df, "M").columns
                ]
            )
        )

    @logTest
    def test_correlation(self) -> None:
        corr = computeCorrelation(self.df)
        self.assertEqual(set(corr.index).difference(corr.columns), set())

        columnName = "TempM"

        corr1 = computeCorrelation(self.df, self.df[[columnName]])[columnName]

        corr2 = computeCorrelation(self.df)[columnName]

        self.assertTrue((corr1 - corr2).sum() < 1e-8)

    @logTest
    def test_flattenMultiIndex(self) -> None:
        cols = ["M", "Pressure"]

        multiIndexDf = pd.concat(
            {k: filterColumnsWith(self.df, k) for k in cols}, axis=1
        )

        self.assertTrue(hasattr(multiIndexDf.columns, "levels"))
        self.assertEqual(len(multiIndexDf.columns.levels), 2)

        flattened_tuple = flattenMultiIndex(multiIndexDf)
        flattened_sep = flattenMultiIndex(multiIndexDf, sep=self.sep_flatten)

        # check indexes tuple and str
        self.assertTrue(all(isinstance(col, tuple) for col in flattened_tuple.columns))
        self.assertTrue(all(isinstance(col, str) for col in flattened_sep.columns))
        self.assertTrue(
            all(
                col_tp == tuple(col_sep.split(self.sep_flatten))
                for col_tp, col_sep in zip(flattened_tuple.columns, flattened_sep.columns)
            )
        )

        self.assertFalse(hasattr(flattened_tuple.columns, "levels"))
        self.assertFalse(hasattr(flattened_sep.columns, "levels"))
        # Check there has been no modification on the previous dataframe
        self.assertTrue(hasattr(multiIndexDf.columns, "levels"))

        unflattened_tuple = generate_multi_index(flattened_tuple)
        unflattened_sep = generate_multi_index(flattened_sep, sep=self.sep_flatten)

        self.assertTrue(hasattr(unflattened_tuple.columns, "levels"))
        self.assertTrue(hasattr(unflattened_sep.columns, "levels"))

        self.assertEqual(unflattened_tuple, multiIndexDf)
        self.assertEqual(unflattened_sep, multiIndexDf)

    @logTest
    def test_timeShift(self) -> None:
        timed = self.df.copy()
        timed.index = [pd.to_datetime(x) for x in timed.index]

        dt = timedelta(days=30)

        lagged = lagSeries(timed, dt)

        self.assertEqual(timed.index.min() + dt, lagged.index.min())
        self.assertEqual(timed.index.max() + dt, lagged.index.max())


if __name__ == "__main__":
    unittest.main()
