"""Random search optimizer."""

import re
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Union

from py4ai.data.model.ml import PandasDataset
from typing_extensions import TypedDict

from py4ai.analytics.ml import PathLike
from py4ai.analytics.ml.core import Estimator, Evaluator, Splitter
from py4ai.analytics.ml.core.optimizer import (
    CheckpointedOptimizer,
    OptimizerOutput,
    Paramset,
)


class RandomElement(TypedDict):
    """Represent a random element via a callable sampler and some parameters."""

    sampler: Callable[..., Union[str, float, int]]
    params: Dict[str, Any]


class OptimizerRandom(CheckpointedOptimizer):
    """Execute a random search for the hyperparameters of an algorithm."""

    def __init__(
        self,
        estimator: Estimator,
        parameter_space: Dict[str, RandomElement],
        evaluator: Evaluator,
        split_strategy: Splitter,
        max_iterations: int = 50,
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

        :param evaluator: evaluator
        :param split_strategy: a splitter class to create folds with a "split" method. Default = None
        :param max_iterations: number of iterations to do on the optimization process. Default = 50
        :param checkpoints: number of iterations to pass before a checkpoint is made. Default = 50
        :param checkpoints_path: path to save checkpoints. Default = None, no checkpoints are saved
        :param checkpoint_refit: if True refits the estimator with the new best parameters after the checkpoint.
        :param checkpoint_overwrite: if True it overwrites the latest checkpoint
        """
        super(OptimizerRandom, self).__init__(
            estimator=estimator.clone(),
            evaluator=evaluator,
            split_strategy=split_strategy,
            checkpoints_path=checkpoints_path,
            checkpoint_refit=checkpoint_refit,
            checkpoint_overwrite=checkpoint_overwrite,
        )

        self._parameter_space = parameter_space

        self._max_iterations = max_iterations
        self._start_iteration = 0
        self._checkpoints = checkpoints

        self._history = {}

        self._run_id = None
        self._output = None

    def __next__(self) -> Paramset:
        """
        Generate new random parameters.

        :return: new random parameters
        """
        return {
            k: v["sampler"](**v["params"])
            for k, v in iter(self._parameter_space.items())
        }

    def set_max_iterations(self, max_iterations: int) -> None:
        """
        Set the number of max iterations.

        :param max_iterations: number of max iterations
        """
        self._max_iterations = max_iterations

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

        self.best_run = self._get_best() if len(self._history) > 0 else None

        start_time = time.time()
        actual_time = start_time

        for iteration in range(self._start_iteration, self._max_iterations):
            new_params_set = self.__next__()
            self.logger.info(
                f"Iteration: {len(self._history) + 1} / {self._max_iterations}"
            )
            self.logger.debug(
                f"Parameters: {';'.join([f'{k}:{v}' for k, v in new_params_set.items()])}"
            )
            sys.stdout.flush()

            previous_time = actual_time
            actual_time = time.time()

            if iteration == 1:
                self.logger.info("======================================")
                self.logger.info(
                    f"Expected duration: {(actual_time-previous_time) * (self._max_iterations-iteration)} seconds"
                )
                self.logger.info(
                    "Expected end time: %s"
                    % (
                        time.strftime(
                            "%Y-%m-%d %H:%M:%S",
                            time.localtime(
                                start_time
                                + (actual_time - previous_time)
                                * (self._max_iterations - iteration)
                            ),
                        )
                    )
                )
                self.logger.info("======================================")
                sys.stdout.flush()
            elif (iteration + 1) % 50 == 0:
                self.logger.info("======================================")
                self.logger.info(
                    f"Expected remaining duration: "
                    f"{(actual_time - previous_time) * (self._max_iterations - iteration)} seconds"
                )
                self.logger.info(
                    "Expected end time: %s"
                    % (
                        time.strftime(
                            "%Y-%m-%d %H:%M:%S",
                            time.localtime(
                                start_time
                                + (actual_time - start_time)
                                * (self._max_iterations * 1.0 / iteration)
                            ),
                        )
                    )
                )
                self.logger.info("======================================")
                sys.stdout.flush()

            if self._check_params_validity(new_params_set):
                _ = self.step(params=new_params_set, dataset=dataset)
                if (iteration != 0) and (iteration % self._checkpoints == 0):
                    self.save_checkpoint(dataset, iteration=iteration)

        return OptimizerOutput(
            model=self.best_model(dataset),
            dataset=dataset,
            pred=self.best_predictions,
            history=self._history,
            predictions=self._predictions if (self.store_predictions is True) else None,
        )
