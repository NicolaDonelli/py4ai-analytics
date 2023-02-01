"""Implementation of the OptimizerStepwiseForward class."""
import re
import sys
from datetime import datetime
from itertools import combinations
from math import factorial
from typing import List, Optional

from py4ai.data.model.ml import PandasDataset

from py4ai.analytics.ml.core import Estimator, Evaluator, Splitter
from py4ai.analytics.ml.core.optimizer import OptimizerOutput
from py4ai.analytics.ml.core.optimizer.feature import FeatureOptimizer
from py4ai.analytics.ml.wrapper.sklearn import ParameterGrid


class OptimizerStepwiseForward(FeatureOptimizer):
    """Execute a forward stepwise selection over possible features given in run's method input dataset."""

    def __init__(
        self,
        estimator: Estimator,
        evaluator: Evaluator,
        split_strategy: Splitter,
        checkpoints_path: Optional[str] = None,
        checkpoint_refit: bool = True,
        checkpoint_overwrite: bool = True,
        max_features: Optional[int] = None,
        min_features: Optional[int] = None,
        features_subselection: Optional[List[str]] = None,
    ):
        """
        Class instance initiliazer.

        :param estimator: estimator or model to optimize
        :param evaluator: evaluator
        :param split_strategy: a splitter class to create folds with a "split" method. Default = None
        :param checkpoints_path: path to save checkpoints. If None (default) no checkpoints are saved.
        :param checkpoint_refit: if True refits the estimator with the new best parameters after the checkpoint. Default = True
        :param checkpoint_overwrite: if True it overwrites the latest checkpoint
        :param max_features: maximum length of columns combination to consider. Exclude combinations of more columns.
        :param min_features: minimum length of columns combination to consider. Exclude combinations of less columns.
        :param features_subselection: Deprecated. Subset of features to keep from the features of the dataset in input
            to run. Instead of setting this parameter the user should instead define only the features he wants for the
            optimization process in the input dataset.
        """
        super(OptimizerStepwiseForward, self).__init__(
            estimator=estimator,
            evaluator=evaluator,
            split_strategy=split_strategy,
            checkpoints_path=checkpoints_path,
            checkpoint_refit=checkpoint_refit,
            checkpoint_overwrite=checkpoint_overwrite,
            max_features=max_features,
            min_features=min_features,
            features_subselection=features_subselection,
        )

    def _initial_vars(self, columns: List[str]) -> ParameterGrid:
        """
        Generate parameter space of all possible combinations of input columns.

        :param columns: list of colnames to combine
        :return: dictionary of the form required by _generate_grid containing all combinations
        """
        initial_combos = list(
            map(
                lambda x: sorted(list(x)),
                combinations(list(columns), self.min_features),
            )
        )

        self.logger.info(f"There are {len(initial_combos)} initial combinations")
        sys.stdout.flush()

        return ParameterGrid({self._par_name: initial_combos})

    def optimize(self, dataset: PandasDataset) -> OptimizerOutput:
        """
        Run the optimization process.

        :param dataset: training dataset

        :raises TypeError: if dataset is not a PandasDataset

        :return: OptimizerOutput with optimization information
        """
        if not isinstance(dataset, PandasDataset):
            raise TypeError("Input dataset must be a PandasDataset")

        red_dataset = self.select_features(dataset)

        self._run_id = "run " + re.sub("[:.]", " ", datetime.today().isoformat())
        self.best_run = self._get_best() if len(self._history) > 0 else None

        n = red_dataset.features.shape[1]
        f = factorial
        k = self.min_features
        m = self.max_features if self.max_features is not None else n

        self.logger.info(
            "Optimization will require up to %d iterations"
            % int(
                f(n)
                / (f(k) * f(n - k))
                * ((n - k) * (n - k + 1) - (n - m - 1) * (n - m))
                / 2
            )
        )

        for iteration, init_paramset in enumerate(
            self._initial_vars(red_dataset.features.columns)
        ):

            self.logger.info("**** Iteration: %d ****" % (iteration + 1))

            tmp_row = self.step(dataset=red_dataset, params=init_paramset)
            best_valid_mean_metric = tmp_row["valid_mean_metric"]
            best_paramset = init_paramset
            tmp_best_paramset = init_paramset

            self.logger.info(
                "Initial validation mean error %.3f with: [%s]"
                % (best_valid_mean_metric, ", ".join(best_paramset[self._par_name]))
            )
            sys.stdout.flush()
            check_increase = True

            while check_increase:
                old_best = best_valid_mean_metric

                self.logger.info(
                    "Length of newly tested paramsets: %d"
                    % (len(best_paramset[self._par_name]) + 1)
                )
                self.logger.info(
                    "Number of newly tested paramsets: %d"
                    % len(
                        set(red_dataset.features.columns).difference(
                            best_paramset[self._par_name]
                        )
                    )
                )
                sys.stdout.flush()
                count = 0
                for col in sorted(
                    set(red_dataset.features.columns).difference(
                        best_paramset[self._par_name]
                    )
                ):
                    count += 1
                    tmp_paramset = {
                        self._par_name: sorted(best_paramset[self._par_name] + [col])
                    }
                    self.logger.info(
                        "%d. Testing [%s]"
                        % (count, ", ".join(tmp_paramset[self._par_name]))
                    )

                    try:
                        tmp_row = self._history[self._get_param_hash(tmp_paramset)]
                        self.logger.info("%d. Reading from history" % count)
                    except KeyError:
                        self.logger.info("%d. Computing step" % count)
                        tmp_row = self.step(dataset=red_dataset, params=tmp_paramset)

                    self.logger.info(
                        "%d. Validation mean error: %.3f"
                        % (count, tmp_row["valid_mean_metric"])
                    )

                    if tmp_row["valid_mean_metric"] < best_valid_mean_metric:
                        best_valid_mean_metric = tmp_row["valid_mean_metric"]
                        tmp_best_paramset = tmp_row["params"]
                        self.logger.info(
                            "Paramset [%s] increases validation mean error to: %.3f"
                            % (
                                ", ".join(tmp_best_paramset[self._par_name]),
                                best_valid_mean_metric,
                            )
                        )
                        sys.stdout.flush()

                best_paramset = tmp_best_paramset
                self.logger.info(
                    "New best validation mean error %.3f obtained with: %s"
                    % (
                        best_valid_mean_metric,
                        ", ".join(best_paramset[self._par_name]),
                    )
                )
                sys.stdout.flush()
                check_increase = (
                    (old_best > best_valid_mean_metric)
                    if len(best_paramset[self._par_name]) < m
                    else False
                )

            if self._checkpoints_path is not None:
                self.save_checkpoint(red_dataset, iteration)

        return OptimizerOutput(
            model=self.best_model(red_dataset),
            dataset=red_dataset,
            pred=self.best_predictions,
            history=self._history,
            predictions=self._predictions if (self.store_predictions is True) else None,
        )
