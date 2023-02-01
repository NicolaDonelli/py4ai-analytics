"""Grid search optimizer."""
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Union

from py4ai.data.model.ml import PandasDataset

from py4ai.analytics.ml import PathLike
from py4ai.analytics.ml.core import Estimator, Evaluator, Splitter
from py4ai.analytics.ml.core.optimizer import CheckpointedOptimizer, OptimizerOutput
from py4ai.analytics.ml.wrapper.sklearn import ParameterGrid
from py4ai.analytics.ml.wrapper.sklearn.splitter import SklearnSplitter
from py4ai.analytics.ml.wrapper.sklearn.wrapper import GroupKFold

ParamValues = List[Union[int, float, str]]


class OptimizerGrid(CheckpointedOptimizer[PandasDataset]):
    """Class to execute a grid search for the hyperparameters of an estimator."""

    def __init__(
        self,
        estimator: Estimator,
        parameter_space: Dict[str, ParamValues],
        evaluator: Evaluator,
        split_strategy: Splitter,
        checkpoints: int = 50,
        checkpoints_path: Optional[PathLike] = None,
        checkpoint_refit: bool = True,
        checkpoint_overwrite: bool = True,
    ) -> None:
        """
        Class instance initializer.

        :param estimator: estimator or model to optimize
        :param parameter_space: dictionary of the form
            {"HyperparName": {"sampling_func":func, "sampling_params":{}, "type": str}}
            for float,int, str hyperparameters sampling_params can be: {"values":[list with values]}
            for float or int parameters sampling_params can also be be:
            {"low": minimum value to consider, "high": maximum value to consider, "type": 'int' or 'float'}}
        :param evaluator: evaluator error metrics to calculate performance.
            # Format {"name of metric": {"function":function,"probas":True or false depending if the metric needs
            # probabilities or not}}
        :param split_strategy: a splitter class to create folds with a "split" method. Default = None
        :param checkpoints: number of iterations to pass before a checkpoint is made. Default = 50.
        :param checkpoints_path: path to save checkpoints. Default = None, no checkpoints are saved.
        :param checkpoint_refit: if True refits the estimator with the new best parameters after the checkpoint.
            Default = True
        :param checkpoint_overwrite: if True it overwrites the latest checkpoint
        """
        super(OptimizerGrid, self).__init__(
            estimator=estimator.clone(),
            evaluator=evaluator,
            split_strategy=split_strategy
            if split_strategy is not None
            else SklearnSplitter(GroupKFold()),
            checkpoints_path=checkpoints_path,
            checkpoint_refit=checkpoint_refit,
            checkpoint_overwrite=checkpoint_overwrite,
        )

        self._parameter_space = parameter_space

        self._checkpoints = checkpoints

        self._output = None
        self._params_grid = ParameterGrid(parameter_space)

    def optimize(self, dataset: PandasDataset) -> OptimizerOutput:
        """
        Start the optimization process.

        :param dataset: training dataset

        :raises TypeError: if dataset is not a PandasDataset

        :return: OptimizerOutput with optimization information
        """
        if not isinstance(dataset, PandasDataset):
            raise TypeError("Input dataset must be a PandasDataset")

        self._run_id = "run " + re.sub("[:.]", " ", datetime.today().isoformat())

        n_max = len(self._params_grid)

        self.best_run = self._get_best() if len(self._history) > 0 else None

        start_time = time.time()
        actual_time = start_time

        for iteration, paramset in enumerate(self._params_grid):

            self.logger.info(f"Iteration: {len(self._history) + 1} / {n_max}")
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
                _ = self.step(params=paramset, dataset=dataset)

                if (iteration != 0) and (iteration % self._checkpoints == 0):
                    self.save_checkpoint(dataset, iteration)

        return OptimizerOutput(
            model=self.best_model(dataset),
            dataset=dataset,
            pred=self.best_predictions,
            history=self._history,
            predictions=self._predictions if (self.store_predictions is True) else None,
        )
