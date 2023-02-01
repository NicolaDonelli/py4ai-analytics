import os
import unittest
import pandas as pd
from py4ai.data.model.ml import PandasTimeIndexedDataset

from py4ai.analytics.ml.core.enricher.transformer.discretizer import Discretizer
from py4ai.core.tests.core import TestCase, logTest
from tests import DATA_FOLDER


class TestDiscretizer(TestCase):
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
    def test_discr(self) -> None:
        discr = Discretizer(
            {
                "features": {
                    "DewPointM": {"threshold": [50, 60], "label_names": [0, 1, 2]}
                },
                "labels": {"TempM": {"threshold": [50]}},
            }
        )
        new_df = discr.transform(self.pdtids)

        self.assertEqual(new_df.features.shape, self.pdtids.features.shape)

        self.assertEqual(new_df.labels.shape, self.pdtids.labels.shape)


if __name__ == "__main__":
    unittest.main()
