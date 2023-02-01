import os
import unittest

import pandas as pd
from py4ai.data.model.ml import PandasTimeIndexedDataset
from py4ai.core.tests.core import TestCase, logTest

from py4ai.analytics.ml.core.enricher.transformer.selector import (
    FeatureSelector,
    LabelSelector,
)
from py4ai.analytics.ml.core.estimator.selector import (
    DeleteConstantFeatures,
    DeleteDuplicates,
)
from py4ai.analytics.ml.core.pipeline.transformer import PipelineEstimator
from py4ai.analytics.ml.wrapper.sklearn.estimator import Estimator as SklearnEstimator
from py4ai.analytics.ml.wrapper.sklearn.wrapper import (
    RandomForestClassifier,
    StandardScaler,
)
from tests import DATA_FOLDER


class PipelineEstimatorTest(TestCase):
    pdtids: PandasTimeIndexedDataset
    pdtidsd: PandasTimeIndexedDataset
    discr_label: PandasTimeIndexedDataset

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(
            os.path.join(DATA_FOLDER, "weather_nyc_short.csv"), index_col="Date"
        )
        cls.pdtids = PandasTimeIndexedDataset(
            features=df.drop("TempM", axis=1), labels=df.TempM
        )
        cls.pdtidsd = PandasTimeIndexedDataset(
            features=df.drop("TempM", axis=1), labels=(df.TempM > 50).map(int)
        )

        df_discr_label = df.copy()
        df_discr_label.TempM = df_discr_label.apply(
            lambda x: 0 if x.TempM < df.TempM.median() else 1, axis=1
        )
        cls.discr_label = PandasTimeIndexedDataset(
            features=df_discr_label.drop("TempM", axis=1),
            labels=df_discr_label.TempM,
        )

    @logTest
    def test_clone(self) -> None:
        pe = PipelineEstimator(
            [("estimator", SklearnEstimator(skclass=RandomForestClassifier()))]
        )
        cloned = pe.clone()

        self.assertNotEqual(id(pe), id(cloned))
        self.assertNotEqual(id(pe.steps[0][1]), id(cloned.steps[0][1]))

    @logTest
    def test_append_step(self) -> None:
        pe = PipelineEstimator(
            [("estimator", SklearnEstimator(skclass=RandomForestClassifier()))]
        )
        self.assertListEqual(
            [
                x[0]
                for x in pe._append_step(
                    pe.steps,
                    ("featureSelector", FeatureSelector(["PressureA", "DewPointM"])),
                )
            ],
            ["estimator", "featureSelector"],
        )

    @logTest
    def test_prepend_step(self) -> None:
        pe = PipelineEstimator(
            [("estimator", SklearnEstimator(skclass=RandomForestClassifier()))]
        )
        self.assertListEqual(
            [
                x[0]
                for x in pe._prepend_step(
                    pe.steps,
                    ("featureSelector", FeatureSelector(["PressureA", "DewPointM"])),
                )
            ],
            ["featureSelector", "estimator"],
        )

    @logTest
    def test_prepend_steps(self) -> None:
        pe = PipelineEstimator(
            [("estimator", SklearnEstimator(skclass=RandomForestClassifier()))]
        )
        self.assertListEqual(
            [
                x[0]
                for x in pe.prepend_steps(
                    [
                        (
                            "featureSelector",
                            FeatureSelector(["PressureA", "DewPointM"]),
                        ),
                        ("labelSelector", LabelSelector(["TempM"])),
                    ]
                ).steps
            ],
            ["featureSelector", "labelSelector", "estimator"],
        )

    @logTest
    def test_prepend(self) -> None:
        pe = PipelineEstimator(
            [("estimator", SklearnEstimator(skclass=RandomForestClassifier()))]
        )
        self.assertListEqual(
            [
                x[0]
                for x in pe.prepend(
                    ("featSel", FeatureSelector(["PressureA", "DewPointM"]))
                ).steps
            ],
            ["featSel", "estimator"],
        )

    @logTest
    def test_append(self) -> None:
        pe = PipelineEstimator(
            [("estimator", SklearnEstimator(skclass=RandomForestClassifier()))]
        )
        self.assertListEqual(
            [
                x[0]
                for x in pe.append(
                    ("featSel", FeatureSelector(["PressureA", "DewPointM"]))
                ).steps
            ],
            ["estimator", "featSel"],
        )

    @logTest
    def test_append_steps(self) -> None:
        pe = PipelineEstimator(
            [("estimator", SklearnEstimator(skclass=RandomForestClassifier()))]
        )
        self.assertListEqual(
            [
                x[0]
                for x in pe.append_steps(
                    [
                        (
                            "featureSelector",
                            FeatureSelector(["PressureA", "DewPointM"]),
                        ),
                        ("labelSelector", LabelSelector(["TempM"])),
                    ]
                ).steps
            ],
            ["estimator", "featureSelector", "labelSelector"],
        )

    @logTest
    def test_insert_before_name(self) -> None:
        pe = PipelineEstimator(
            [
                ("featSel", FeatureSelector(to_keep=["PressureA", "DewPointM"])),
                ("estimator", SklearnEstimator(skclass=RandomForestClassifier())),
            ]
        )
        self.assertListEqual(
            [
                x[0]
                for x in pe.insert_before_name(
                    name="featSel", step=("labSel", LabelSelector(to_keep=["TempM"]))
                ).steps
            ],
            ["labSel", "featSel", "estimator"],
        )

    @logTest
    def test_insert_after_name(self) -> None:
        pe = PipelineEstimator(
            [
                ("featSel", FeatureSelector(to_keep=["PressureA", "DewPointM"])),
                ("estimator", SklearnEstimator(skclass=RandomForestClassifier())),
            ]
        )
        self.assertListEqual(
            [
                x[0]
                for x in pe.insert_after_name(
                    name="featSel", step=("labSel", LabelSelector(to_keep=["TempM"]))
                ).steps
            ],
            ["featSel", "labSel", "estimator"],
        )

    @logTest
    def test_pipe_of_pipe(self) -> None:
        pe = PipelineEstimator(
            [
                (
                    "step1",
                    PipelineEstimator(
                        [
                            (
                                "feature_selector",
                                FeatureSelector(to_keep=["PressureA", "DewPointM"]),
                            ),
                            (
                                "estimator",
                                SklearnEstimator(skclass=RandomForestClassifier()),
                            ),
                        ]
                    ),
                )
            ]
        )

        self.assertListEqual(
            [x[0] for x in pe.steps], ["step1_feature_selector", "step1_estimator"]
        )

    @logTest
    def test_pipe_of_pipes(self) -> None:
        pe = PipelineEstimator(
            [
                (
                    "step1",
                    PipelineEstimator(
                        [
                            (
                                "feature_selector",
                                FeatureSelector(to_keep=["PressureA", "DewPointM"]),
                            ),
                            ("label_selector", LabelSelector(to_keep=["TempM"])),
                        ]
                    ),
                ),
                ("step2", DeleteConstantFeatures()),
                (
                    "step3",
                    PipelineEstimator(
                        [
                            ("deleter", DeleteDuplicates()),
                            (
                                "estimator",
                                SklearnEstimator(skclass=RandomForestClassifier()),
                            ),
                        ]
                    ),
                ),
            ]
        )

        self.assertListEqual(
            [x[0] for x in pe.steps],
            [
                "step1_feature_selector",
                "step1_label_selector",
                "step2",
                "step3_deleter",
                "step3_estimator",
            ],
        )

    @logTest
    def test_train_pipe_of_pipes(self) -> None:
        pe = PipelineEstimator(
            [
                (
                    "step1",
                    PipelineEstimator(
                        [
                            (
                                "feature_selector",
                                FeatureSelector(to_keep=["PressureA", "DewPointM"]),
                            ),
                            ("label_selector", LabelSelector(to_keep=["TempM"])),
                        ]
                    ),
                ),
                ("step2", DeleteConstantFeatures()),
                (
                    "step3",
                    PipelineEstimator(
                        [
                            ("deleter", DeleteDuplicates()),
                            (
                                "estimator",
                                SklearnEstimator(skclass=RandomForestClassifier()),
                            ),
                        ]
                    ),
                ),
            ]
        )
        pm = pe.train(self.pdtidsd)

        self.assertListEqual(
            [x[1].estimator for x in pm.steps],
            [None, None, pe.steps[2][1], pe.steps[3][1], pe.steps[4][1]],
        )

    @logTest
    def test_add(self) -> None:
        pe1 = PipelineEstimator(
            [
                (
                    "feature_selector",
                    FeatureSelector(to_keep=["PressureA", "DewPointM"]),
                ),
                ("label_selector", LabelSelector(to_keep=["TempM"])),
            ]
        )

        pe2 = PipelineEstimator(
            [
                ("deleter", DeleteDuplicates()),
                ("estimator", SklearnEstimator(skclass=RandomForestClassifier())),
            ]
        )
        pe = pe1 + pe2
        self.assertListEqual(
            [x[0] for x in pe.steps],
            ["feature_selector", "label_selector", "deleter", "estimator"],
        )
        self.assertListEqual(
            [x[1] for x in pe.steps],
            [pe1.steps[0][1], pe1.steps[1][1], pe2.steps[0][1], pe2.steps[1][1]],
        )

    @logTest
    def test_as_pipeline(self) -> None:
        est = SklearnEstimator(skclass=RandomForestClassifier())
        pe = PipelineEstimator(steps=[("estimator", est)])
        self.assertEqual(pe.steps[0][0], "estimator")
        self.assertEqual(pe.steps[0][1], est)

    @logTest
    def test_predict_proba_PipelineEstimator(self) -> None:
        rfs = [
            ("standard", StandardScaler()),
            ("rf", RandomForestClassifier(random_state=42, n_estimators=100)),
        ]

        mod = PipelineEstimator(
            [(name, SklearnEstimator(model)) for name, model in rfs]
        )

        model_trained = mod.train(self.discr_label)

        predict_proba = model_trained.transform(self.discr_label)

        self.assertGreater(len(predict_proba), 0)


if __name__ == "__main__":
    unittest.main()
