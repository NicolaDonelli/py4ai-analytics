import numpy as np
import pandas as pd
from py4ai.core.logging import WithLogging
from py4ai.analytics.ml.core import FeatureProcessing, TimeSeriesFeatureProcessing
from py4ai.core.utils.decorators import lazyproperty
from py4ai.analytics.ml.core.splitter.time_evolving import TimeEvolvingSplitter
from py4ai.analytics.ml.wrapper.sklearn.evaluator import (
    RegressionScorer,
    mean_absolute_error,
    root_mean_squared_error,
)


class CustomReport(RegressionScorer):
    base_evaluators = {"RMSE": root_mean_squared_error, "MAE": mean_absolute_error}


class TimeEvolvingTest(WithLogging):
    """Function to test models using TimeEvolvingFold"""

    def __init__(
        self,
        step,
        test_start=None,
        nmax=np.inf,
        window=None,
        g=lambda x: x.date(),
        prob=False,
    ):
        """
        :param step: number of periods to include on each fold.
        :param nmax: maximum number of folds. Default = np.inf
        :param window: length of fixed window to use for the training set. Default = None
        :param test_start: start date for the validation set. Default = TEST_START
        :param g: grouping function. Default = lambda x: x.date()
        :param prob: if true it uses predict_proba to create predictions. Default = False

        :type step: int
        :type nmax: int
        :type window: int
        :type test_start: timestamp
        :type g: function
        :type prob: bool
        """

        self.step = step
        self.nmax = nmax
        self.window = window
        self.test_start = test_start
        self.g = g
        self.prob = prob

    def run(self, model, train, test):
        """
        Method to get a dataframe with predictions using a TimeEvolving Logic

        :param model: model to use on the testing
        :param train: train dataset
        :param test: test dataset

        :type model: sklearn.Pipeline, model
        :type train: Dataset
        :type test: Dataset

        :return: Dataframe with real values and their respective predictions or probabilities in case of prob=True
        :rtype: pd.DataFrame
        """

        test_start = self.test_start if (self.test_start) else test.labels.index.min()

        pipeline = model.clone()
        tester = TimeEvolvingSplitter(
            min_periods_per_fold=self.step,
            n_folds=self.nmax,
            valid_start=test_start,
            window=self.window,
            g=self.g,
        )

        dataset = train.union(test)

        predictions = []

        for _train, _test in tester.split(dataset):

            self.logger.info(
                "Dates tested: %s"
                % ", ".join(
                    np.sort(np.unique([str(x.date()) for x in _test.features.index]))
                )
            )

            fitted = pipeline.train(_train)
            preds_test = fitted.transform(_test)

            predictions.append((preds_test.labels.iloc[:, 0], preds_test.features))

        true = pd.concat([x[0] for x in predictions])
        pred = pd.concat([x[1] for x in predictions])

        if not self.prob:
            return pd.concat({"true": true, "pred": pred}, axis=1)
        else:
            return pd.concat([pd.DataFrame({"true": true}), pred], axis=1)


class FeatureProcessingTimeSeriesExample(TimeSeriesFeatureProcessing):
    @property
    def frequency(self) -> str:
        return "D"

    @lazyproperty
    def df(self) -> pd.DataFrame:
        return self.input_source.set_index(
            pd.to_datetime(self.input_source.index.tolist())
        )

    def feature_space(self, range=None) -> pd.DataFrame:
        df_features = self.df.drop("TempM", axis=1)
        return df_features if range is None else df_features.loc[range]

    def labels(self, range=None) -> pd.DataFrame:
        df_labels = self.df["TempM"].to_frame("pred")
        return df_labels if range is None else df_labels.loc[range]


class FeatureProcessingExample(FeatureProcessing):
    @lazyproperty
    def df(self) -> pd.DataFrame:
        return self.input_source.reset_index()

    def feature_space(self, range=None) -> pd.DataFrame:
        df_features = self.df.drop("TempM", axis=1)
        return df_features if range is None else df_features.loc[range]

    def labels(self, range=None) -> pd.DataFrame:
        df_labels = self.df[["TempM"]]
        return df_labels if range is None else df_labels.loc[range]
