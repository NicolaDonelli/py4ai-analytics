"""Implementation of the OptimizerFeatureSearch class."""

import re
import sys
import time
from datetime import datetime
from itertools import chain, combinations
from typing import List, Optional

import numpy as np
from py4ai.data.model.ml import PandasDataset

from py4ai.analytics.ml import PathLike
from py4ai.analytics.ml.core import Estimator, Evaluator, Splitter
from py4ai.analytics.ml.core.optimizer import OptimizerOutput
from py4ai.analytics.ml.core.optimizer.feature import FeatureOptimizer
from py4ai.analytics.ml.wrapper.sklearn import ParameterGrid


class OptimizerFeatureSearch(FeatureOptimizer):
    """Execute a grid search for the set of features to use to train the input estimator."""

    def __init__(
        self,
        estimator: Estimator,
        evaluator: Evaluator,
        split_strategy: Optional[Splitter] = None,
        max_combinations: Optional[int] = None,
        max_features: Optional[int] = None,
        min_features: Optional[int] = None,
        features_subselection: Optional[List[str]] = None,
        checkpoints_path: Optional[PathLike] = None,
        checkpoints: int = 50,
        checkpoint_refit: bool = True,
        checkpoint_overwrite: bool = True,
    ):
        """
        Class instance initializer.

        :param estimator: estimator or model to optimize
        :param evaluator: evaluator
        :param split_strategy: a splitter class to create folds with a "split" method. Default = None
        :param max_combinations: maximum number of columns combination to consider.
        :param max_features: maximum length of columns combination to consider. Exclude combinations of more columns.
        :param min_features: minimum length of columns combination to consider. Exclude combinations of less columns.
        :param features_subselection: Deprecated. Subset of features to keep from the features of the dataset in input
            to run. Instead of setting this parameter the user should instead define only the features he wants for the
            optimization process in the input dataset.
        :param checkpoints: number of iterations to pass before a checkpoint and a time estimation is made. Default = 50
        :param checkpoints_path: path to save checkpoints. If None (default) no checkpoints are saved.
        :param checkpoint_refit: if True refits the estimator with the new best parameters after the checkpoint.
            Default = True
        :param checkpoint_overwrite: if True it overwrites the latest checkpoint
        """
        super(OptimizerFeatureSearch, self).__init__(
            estimator=estimator,
            evaluator=evaluator,
            split_strategy=split_strategy,
            max_features=max_features,
            min_features=min_features,
            features_subselection=features_subselection,
            checkpoints_path=checkpoints_path,
            checkpoint_refit=checkpoint_refit,
            checkpoint_overwrite=checkpoint_overwrite,
        )

        self._checkpoints = checkpoints
        self.max_combinations = max_combinations

    def _generate_parameter_grid(self, columns: List[str]) -> ParameterGrid:
        """
        Generate parameter space of all possible combinations of input columns.

        :param columns: list of colnames to combine
        :return: dictionary of the form required by _generate_grid containing all combinations
        """

        def powerset(iterable, min_len=None, max_len=None):
            s = list(iterable)
            min_len = min_len if min_len is not None else 1
            max_len = max_len if max_len is not None else len(s)
            return chain.from_iterable(
                combinations(s, r) for r in range(min_len, max_len + 1)
            )

        self.logger.info("Generating combinations of features")
        sys.stdout.flush()

        comb = list(
            map(
                list,
                powerset(columns, min_len=self.min_features, max_len=self.max_features),
            )
        )

        if self.max_combinations is not None:
            max_combinations = (
                self.max_combinations
                if self.max_combinations <= len(comb)
                else len(comb)
            )
            comb = list(np.random.choice(comb, size=max_combinations, replace=False))  # type: ignore

        return ParameterGrid({self._par_name: comb})

    def optimize(self, dataset: PandasDataset) -> OptimizerOutput:
        """
        Run the optimization process.

        :param dataset: training dataset

        :raises TypeError: if dataset is not a PandasDatase

        :return: OptimizerOutput with optimization information
        """
        if not isinstance(dataset, PandasDataset):
            raise TypeError("Input dataset must be a PandasDataset")

        red_dataset = self.select_features(dataset)

        self._run_id = "run " + re.sub("[:.]", " ", datetime.today().isoformat())

        self._params_grid: ParameterGrid = self._generate_parameter_grid(red_dataset.features.columns)

        n_max = len(self._params_grid)

        self.best_run = self._get_best() if len(self._history) > 0 else None

        start_time = time.time()
        actual_time = start_time

        for iteration, paramset in enumerate(self._params_grid):

            self.logger.info("Iteration: %d / %d" % (len(self._history) + 1, n_max))
            self.logger.debug(
                "Parameters: %s"
                % ";".join(["%s:%s" % (k, v) for k, v in paramset.items()])
            )
            sys.stdout.flush()

            previous_time = actual_time
            actual_time = time.time()

            if iteration == 1:
                self.logger.info("======================================")
                self.logger.info(
                    "Expected duration: %f seconds"
                    % ((actual_time - previous_time) * (n_max - iteration))
                )
                self.logger.info(
                    "Expected end time: %s"
                    % (
                        time.strftime(
                            "%Y-%m-%d %H:%M:%S",
                            time.localtime(
                                start_time
                                + (actual_time - previous_time) * (n_max - iteration)
                            ),
                        )
                    )
                )
                self.logger.info("======================================")
                sys.stdout.flush()
            elif (iteration + 1) % self._checkpoints == 0:
                self.logger.info("======================================")
                self.logger.info(
                    "Expected remaining duration: %f seconds"
                    % ((actual_time - previous_time) * (n_max - iteration))
                )
                self.logger.info(
                    "Expected end time: %s"
                    % (
                        time.strftime(
                            "%Y-%m-%d %H:%M:%S",
                            time.localtime(
                                start_time
                                + (actual_time - start_time) * (n_max * 1.0 / iteration)
                            ),
                        )
                    )
                )
                self.logger.info("======================================")
                sys.stdout.flush()

            if self._check_params_validity(paramset):
                _ = self.step(params=paramset, dataset=red_dataset)

                if (
                    (self._checkpoints_path is not None)
                    and (iteration != 0)
                    and (iteration % self._checkpoints == 0)
                ):
                    self.save_checkpoint(red_dataset, iteration)

        return OptimizerOutput(
            model=self.best_model(red_dataset),
            dataset=red_dataset,
            pred=self.best_predictions,
            history=self._history,
            predictions=self._predictions if (self.store_predictions is True) else None,
        )
