"""Read modelling process outputs."""
import os

import pandas as pd
from py4ai.data.model.ml import PandasDataset
from py4ai.core.logging import WithLogging
from py4ai.core.utils.decorators import lazyproperty as lazy

from py4ai.analytics.ml.core import FeatureProcessing, Transformer


class RunnerResults(WithLogging):
    # TODO: the module should be completely reengineered accordingly to issue #6
    """Class to retrieve results coming from a Model Run."""

    def __init__(self, path: str) -> None:
        """
        Initialize the class.

        :param path: path to the folder where the Model Run are stored
        """
        self.path = path

    @lazy
    def model(self) -> Transformer:
        """
        Load best model.

        Best model identified in the Model Run.

        :return: the best model (Transformer)
        """
        return Transformer.load(os.path.join(self.path, "model.p"))

    @lazy
    def proc(self) -> FeatureProcessing:
        """
        Load Feature Processing.

        Feature processing used to create train and test sets.

        :return: feature processign (FeatureProcessing)
        """
        return FeatureProcessing.load(os.path.join(self.path, "proc.p"))

    @lazy
    def train_set(self) -> PandasDataset:
        """
        Load training dataset.

        Training set used in the Model Run.

        :return: Dataset of train
        """
        return PandasDataset.load(filename=os.path.join(self.path, "train"))

    @property
    def test_set(self) -> PandasDataset:
        """
        Load test dataset.

        Test set used in the Model Run

        :return: Dataset of test
        """
        return PandasDataset.load(filename=os.path.join(self.path, "test"))

    def get_model(self, key: int) -> Transformer:
        """
        Get model trained over the entire training set corresponding to the hash given by key.

        :param key: hash of the hyper-parameters to be used

        :return: Trained model
        """
        params = self.logResult(
            lambda x: "Training model with params: %s" % str(x), "DEBUG"
        )(self.history.loc[key]["params"])
        return self.model.estimator.set_params(**params).train(self.train_set)

    @staticmethod
    def _getIndexLevels(df: pd.DataFrame) -> int:
        """
        Return the levels number of a dataframe index.

        If error return 1.

        :param df: pandas dataframe
        :return: number of index levels
        """
        try:
            foldingNumber = len(df.index.levels)
        except AttributeError:
            foldingNumber = 1
        return foldingNumber

    @lazy
    def has_folding_number(self) -> bool:
        """
        Whether or not the result has a folding number.

        :return: whether or not the result has a folding number.
        """
        top_run = self.top_runs(1)[0]
        return self._getIndexLevels(self.train_set.features) < self._getIndexLevels(
            self.predictions[top_run]
        )

    @lazy
    def predictions(self) -> pd.DataFrame:
        """
        Retrieve predictions dataframe.

        :return: predictions
        """
        return pd.read_pickle(os.path.join(self.path, "predictions"))

    def get_predictions(self, key: int) -> pd.Series:
        """
        Get predictions on the validation set corresponding to the hash given by key.

        :param key: hash of the hyper-parameters to be used

        :return: Predictions over validation set
        """
        if self.has_folding_number is True:
            return self.predictions[key].droplevel(level=0)
        else:
            return self.predictions[key]

    @lazy
    def history(self) -> pd.DataFrame:
        """
        History of the model run.

        :return: History
        """
        return pd.DataFrame(pd.read_pickle(os.path.join(self.path, "history"))).T

    def top_runs(self, n=None) -> pd.Index:
        """
        Hash of the top runs.

        :param n: number of runs to return

        :return: List of int hashes
        """
        return self.history.sort_values("valid_mean_metric", ascending=False).index[:n]
