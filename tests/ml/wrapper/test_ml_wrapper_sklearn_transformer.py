from py4ai.analytics.ml.wrapper.sklearn.transformer import (
    SelectByMutualInfo,
    SelectByFtest,
)
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    SelectFpr,
)
from py4ai.data.model.ml import PandasDataset
import pandas as pd
import unittest
import numpy as np
from itertools import compress


features = ["f_1", "f_2", "f_3"]
label = "label"

X_c = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
y_c = [1, 0, 1, 1, 1, 0]
df_c = PandasDataset(
    features=pd.DataFrame(data=X_c, columns=features),
    labels=pd.Series(data=y_c, name=label),
)

X_r = np.random.rand(100, 3)
y_r = X_r[:, 0] + np.sin(6 * np.pi * X_r[:, 1]) + 0.1 * np.random.randn(100)
df_r = PandasDataset(
    features=pd.DataFrame(data=X_r, columns=features),
    labels=pd.Series(data=y_r, name=label),
)


class TestSelectByMutualInfo(unittest.TestCase):
    def setUp(self) -> None:
        self.estimator_classif = SelectByMutualInfo("Classification")
        self.estimator_regression = SelectByMutualInfo("Regression")

    def test_train(self):
        mi_c = mutual_info_classif(
            X_c,
            y_c,
            discrete_features=self.estimator_classif.discrete_features,
            n_neighbors=self.estimator_classif.n_neighbors,
            random_state=42,
        )

        mi_c /= np.max(mi_c) * 1.0
        to_keep = [
            i for i, k in zip(features, mi_c) if k > self.estimator_classif.mi_thresh
        ]
        feature_selector_c = self.estimator_classif.train(df_c)
        self.assertEqual(sorted(feature_selector_c.to_keep), sorted(to_keep))

        mi_r = mutual_info_regression(
            X_r,
            y_r,
            discrete_features=self.estimator_regression.discrete_features,
            n_neighbors=self.estimator_regression.n_neighbors,
            random_state=42,
        )

        mi_r /= np.max(mi_r) * 1.0
        to_keep = [
            i for i, k in zip(features, mi_r) if k > self.estimator_regression.mi_thresh
        ]
        feature_selector_r = self.estimator_regression.train(df_r)

        self.assertEqual(sorted(feature_selector_r.to_keep), sorted(to_keep))


class TestSelectByFtest(unittest.TestCase):
    def setUp(self) -> None:
        self.estimator_classif = SelectByFtest("Classification", alpha=0.5)
        self.estimator_regression = SelectByFtest("Regression", alpha=0.5)

    def test_train(self):
        sfpr_c = SelectFpr(
            score_func=lambda x, y: self.estimator_classif.func(x, y),
            alpha=self.estimator_classif.alpha,
        ).fit(X_c, y_c)

        to_keep_c = list(compress(features, sfpr_c.get_support()))
        feature_selector_c = self.estimator_classif.train(df_c)

        self.assertEqual(sorted(feature_selector_c.to_keep), sorted(to_keep_c))

        sfpr_r = SelectFpr(
            score_func=lambda x, y: self.estimator_regression.func(x, y),
            alpha=self.estimator_regression.alpha,
        ).fit(X_r, y_r)
        to_keep_r = list(compress(features, sfpr_r.get_support()))
        feature_selector_r = self.estimator_regression.train(df_r)
        self.assertEqual(sorted(feature_selector_r.to_keep), sorted(to_keep_r))


if __name__ == "__main__":
    unittest.main()
