from typing import Optional
import pandas as pd
from sklearn.metrics import accuracy_score

from py4ai.data.model.ml import Dataset, PandasDataset
from py4ai.analytics.ml.core import Estimator, Transformer
from py4ai.analytics.ml.core.metric import ScoreMetric
from py4ai.analytics.ml.core.metric.coherence import (
    M_cos_nlr,
    M_d,
    M_lc,
    M_lr,
    M_nlr,
    SegmentationMeasure,
    SegmentationMeasureSet,
)
from py4ai.analytics.ml.wrapper.sklearn.evaluator import (
    PandasEvaluator,
    RegressionScorer,
    mean_absolute_error,
    root_mean_squared_error,
)


class CustomReport(RegressionScorer):
    @staticmethod
    def acc(true: pd.Series, pred: pd.Series):
        t = true.apply(lambda x: 1 if x > 0 else -1)
        pred = pred.apply(lambda x: -1 if x < 0 else (1 if x > 0 else 0))

        return accuracy_score(t.loc[pred.index][pred != 0], pred[pred != 0])

    base_evaluators = {
        "RMSE": root_mean_squared_error,
        "MAE": mean_absolute_error,
        "accuracy": PandasEvaluator(acc.__func__, ScoreMetric),
    }


class FakeTrasformer(Transformer):
    """
    Default trasformer that do nothing
    """

    def apply(self, dataset: PandasDataset):
        new_dataset = dataset.createObject(
            labels=dataset.labels, features=pd.Series(["textpurified"] * len(dataset))
        )
        return new_dataset


class MaxCombiner(Transformer):
    def estimator(self) -> Optional[Estimator]:
        return None

    def predict_proba(self, dataset: Dataset) -> pd.DataFrame:
        return self.transform(dataset).getFeaturesAs("pandas")

    def apply(self, dataset: Dataset) -> PandasDataset:
        return PandasDataset(
            self.combine(dataset.getFeaturesAs("pandas")), dataset.getLabelsAs("pandas")
        )

    @staticmethod
    def combine(preds: pd.DataFrame) -> pd.DataFrame:
        return preds.groupby(level=1, axis="columns").max()


allSegmentationMeasureSet = SegmentationMeasureSet(
    {
        SegmentationMeasure("S_one_set", M_cos_nlr()),
        SegmentationMeasure("S_one_any", M_cos_nlr()),
        SegmentationMeasure("S_one_pre", M_cos_nlr()),
        SegmentationMeasure("S_one_all", M_cos_nlr()),
        SegmentationMeasure("S_one_one", M_cos_nlr()),
        SegmentationMeasure("S_one_set", M_d()),
        SegmentationMeasure("S_one_any", M_d()),
        SegmentationMeasure("S_one_pre", M_d()),
        SegmentationMeasure("S_one_all", M_d()),
        SegmentationMeasure("S_one_one", M_d()),
        SegmentationMeasure("S_one_set", M_nlr()),
        SegmentationMeasure("S_one_any", M_nlr()),
        SegmentationMeasure("S_one_pre", M_nlr()),
        SegmentationMeasure("S_one_all", M_nlr()),
        SegmentationMeasure("S_one_one", M_nlr()),
        SegmentationMeasure("S_one_set", M_lr()),
        SegmentationMeasure("S_one_any", M_lr()),
        SegmentationMeasure("S_one_pre", M_lr()),
        SegmentationMeasure("S_one_all", M_lr()),
        SegmentationMeasure("S_one_one", M_lr()),
        SegmentationMeasure("S_one_set", M_lc()),
        SegmentationMeasure("S_one_any", M_lc()),
        SegmentationMeasure("S_one_pre", M_lc()),
        SegmentationMeasure("S_one_all", M_lc()),
        SegmentationMeasure("S_one_one", M_lc()),
    }
)
