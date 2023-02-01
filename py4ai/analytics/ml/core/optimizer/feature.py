"""Implementation of the FeatureOptimizer class."""

from typing import List, Optional

from py4ai.data.model.ml import PandasDataset
from py4ai.core.types import PathLike

from py4ai.analytics.ml.core import Estimator, Evaluator, Splitter
from py4ai.analytics.ml.core.estimator.subselector import SubselectionEstimator
from py4ai.analytics.ml.core.optimizer import CheckpointedOptimizer
from py4ai.analytics.ml.core.splitter.time_evolving import TimeEvolvingSplitter


class FeatureOptimizer(CheckpointedOptimizer[PandasDataset]):
    """Optimize the features."""

    def __init__(
        self,
        estimator: Estimator,
        evaluator: Evaluator,
        split_strategy: Optional[Splitter] = None,
        max_features: Optional[int] = None,
        min_features: Optional[int] = None,
        features_subselection: Optional[List[str]] = None,
        checkpoints_path: Optional[PathLike] = None,
        checkpoint_refit: bool = True,
        checkpoint_overwrite: bool = True,
    ):
        """
        Class instance initializer.

        :param estimator: estimator or model to optimize
        :param evaluator: evaluator
        :param split_strategy: a splitter class to create folds with a "split" method. Default = None
        :param max_features: maximum length of columns combination to consider. Exclude combinations of more columns.
        :param min_features: minimum length of columns combination to consider. Exclude combinations of less columns.
        :param features_subselection: Deprecated. Subset of features to keep from the features of the dataset in input
            to run. Instead of setting this parameter the user should instead define only the features he wants for the
            optimization process in the input dataset.
        :param checkpoints_path: path to save checkpoints. If None (default) no checkpoints are saved.
        :param checkpoint_refit: if True refits the estimator with the new best parameters after the checkpoint.
            Default = True
        :param checkpoint_overwrite: if True it overwrites the latest checkpoint
        """
        super(FeatureOptimizer, self).__init__(
            estimator=SubselectionEstimator(estimator.clone()),
            evaluator=evaluator,
            split_strategy=split_strategy
            if split_strategy is not None
            else TimeEvolvingSplitter(),
            checkpoints_path=checkpoints_path,
            checkpoint_overwrite=checkpoint_overwrite,
            checkpoint_refit=checkpoint_refit,
        )

        self._par_name = "features_name"

        if features_subselection is not None:
            DeprecationWarning(
                "Setting features_subselection is deprecated. The user should define only the features he "
                "wants for the optimization process in the input dataset."
            )

        self._history = {}

        self.features_subselection = features_subselection
        self.min_features = min_features if min_features is not None else 1
        self.max_features = max_features

        self._run_id = None
        self._output = None
        self._params_grid = None

    def select_features(self, dataset: PandasDataset):
        """
        Create new dataset with a subselection of features specified by `features_subselection`.

        :param dataset: input dataset

        :return: dataset with selected features
        """
        return (
            dataset.createObject(
                dataset.features[self.features_subselection], dataset.labels
            )
            if self.features_subselection is not None
            else dataset
        )
