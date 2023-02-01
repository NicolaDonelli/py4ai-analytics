from py4ai.analytics.ml.eda.general import crosscorr
import os
import unittest
from scipy.stats import pearsonr
from py4ai.core.tests.core import TestCase, logTest

import pandas as pd

from tests import DATA_FOLDER


class TestEDAGeneral(TestCase):
    df: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.read_csv(
            os.path.join(DATA_FOLDER, "weather_nyc_short.csv"), index_col="Date"
        )

    @logTest
    def test_crosscorr(self):
        corr, pvalue = pearsonr(self.df.Humidity, self.df.TempM)
        self.assertEqual(crosscorr(self.df.TempM, self.df.TempM)["Corr"][0], 1.0)
        self.assertAlmostEqual(crosscorr(self.df.Humidity, self.df.TempM)["Corr"][0], corr)
        self.assertAlmostEqual(
            crosscorr(self.df.Humidity, self.df.TempM)["Corr"][0], corr
        )
        self.assertAlmostEqual(
            crosscorr(self.df.Humidity, self.df.TempM, pvalflag=True)["Pval"][0], pvalue
        )


if __name__ == "__main__":
    unittest.main()
