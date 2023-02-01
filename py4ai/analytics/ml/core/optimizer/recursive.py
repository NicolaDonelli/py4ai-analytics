"""Recursive optimizer: an efficient version of grid search optimizer for pipelines."""
import sys
import time
from math import ceil
from typing import Dict, Iterable, Iterator, Optional, Tuple, Union, cast

import pandas as pd
from py4ai.data.model.ml import PandasDataset
from py4ai.core.utils.dict import groupIterable, union
from typing_extensions import TypedDict

from py4ai.analytics.ml import PathLike
from py4ai.analytics.ml.core import (
    Estimator,
    Evaluator,
    Resampler,
    Splitter,
    Transformer,
)
from py4ai.analytics.ml.core.optimizer import (
    CheckpointedOptimizer,
    OptimizerOutput,
    Paramset,
)
from py4ai.analytics.ml.core.optimizer.gridsearch import ParamValues
from py4ai.analytics.ml.core.pipeline.transformer import PipelineEstimator
from py4ai.analytics.ml.core.splitter.index_based import IndexSplitter
from py4ai.analytics.ml.wrapper.sklearn import ParameterGrid


class RestrictedParams(TypedDict):
    """Represent the conditions on the restricted parameters."""

    restricted_params: Dict[str, str]


ConditionalParamspace = Dict[str, Union[ParamValues, RestrictedParams]]


class RecursiveOptimizer(CheckpointedOptimizer):
    """Execute a grid search for the hyperparameters of an algorithm optimized for Pipelines."""

    def __init__(
        self,
        estimator: PipelineEstimator,
        parameter_space: ConditionalParamspace,
        evaluator: Evaluator,
        split_strategy: Splitter,
        checkpoints_path: Optional[PathLike] = None,
        checkpoints: int = 50,
        checkpoint_refit: bool = True,
        checkpoint_overwrite: bool = True,
        iteration_info: Optional[Tuple[int, int]] = None,
    ):
        """
        Class instance initializer.

        :param estimator: estimator to optimize
        :param parameter_space: dictionary of the form
            {"HyperparName":[list with values]
            ...
            "condition": {"restricted params": {"stepname__HyperparamName": "PreviousStepname__HyperparamName"}}}
        :param evaluator: evaluator
        :param split_strategy: a splitter class to create folds with a "split" method. Default = None
        :param checkpoints: number of iterations to pass before a checkpoint is made. Default = 50
        :param checkpoints_path: path to save checkpoints. Default None, no checkpoints are saved
        :param checkpoint_refit: if True refits the estimator with the new best parameters after the checkpoint.
            Default = True
        :param checkpoint_overwrite: if True it overwrites the latest checkpoint
        :param iteration_info: Info from the total number of iterations to be done, including outer loops.
            If None means this is the first recursive stage
        """
        super(RecursiveOptimizer, self).__init__(
            estimator=estimator.clone(),
            evaluator=evaluator,
            split_strategy=split_strategy,
            checkpoints_path=checkpoints_path,
            checkpoint_refit=checkpoint_refit,
            checkpoint_overwrite=checkpoint_overwrite,
        )

        self._parameter_space = parameter_space

        self.iteration_info = iteration_info
        self._checkpoints = checkpoints

        self.actual_time: float = time.time()
        self.start_time = time.time()
        self.counter = iteration_info[0] if iteration_info is not None else 0
        self.iteration = (0, 1)

    def get_params_subset(
        self,
        step_name: str,
        diff: bool = False,
        elem: Optional[Dict[str, Union[int, float, str]]] = None,
    ) -> ConditionalParamspace:
        """
        Return dictionary of params.

        :param step_name: name of the current step in the Pipeline
        :param diff: if diff is False, the method considers only the params related to the current step,
            otherwise it exclude them and considers all the others.
        :param elem: dictionary with params used in the current step
        :return: dictionary with params
        """
        # TODO remove cast and identify the right type for parameter_space variable

        cond = (
            cast(RestrictedParams, self._parameter_space["condition"])["restricted_params"]
            if "condition" in self._parameter_space
            else None
        )

        if len(self._parameter_space) == 0:
            return {}

        diz = (
            {k: v for k, v in self._parameter_space.items() if step_name not in k}
            if diff
            else union(
                {
                    k[len(step_name) + 2 :]: v
                    for k, v in self._parameter_space.items()
                    if step_name in k
                },
                {
                    "condition": {
                        "restricted_params": {
                            k_cond.split("__")[-1]: v_cond.split("__")[-1]
                            for k_cond, v_cond in cond.items()
                            if step_name == k_cond.split("__")[0]
                            and step_name == v_cond.split("__")[0]
                        }
                    }
                }
                if cond is not None
                else {},
            )
        )

        if (
            cond is not None and elem is not None
        ):  # condizioni su parametri di step differenti
            for k, v in cond.items():
                if (
                    v.split("__")[-1] in elem.keys()
                    and k.split("__")[0] != v.split("__")[0]
                    and k.split("__")[0] != step_name
                ):
                    diz[k] = [elem[v.split("__")[-1]]]
                else:
                    diz["condition"] = {"restricted_params": cond}
        return diz

    @staticmethod
    def _generate_grid(parameter_space: ConditionalParamspace) -> ParameterGrid:
        """
        Generate the grid from a parameter_space dictionary.

        :param parameter_space: dictionary of the form
            {"HyperparName":[list with values]
            ...
            "condition": {"restricted params": {"stepname__HyperparamName": "PreviousStepname__HyperparamName"}}}

        :return: grid with values to use for each iteration
        """
        to_grid_dict = {k: v for k, v in parameter_space.items() if k != "condition"}

        if "condition" in parameter_space:
            new_dict = {}
            i = 1
            for k, v in cast(RestrictedParams, parameter_space["condition"])["restricted_params"].items():
                if v in parameter_space.keys():
                    list_possible_values = to_grid_dict[v]
                    new_dict[str(i)] = [
                        {k: elem, v: elem} for elem in list_possible_values
                    ]
                    i += 1

            if len(new_dict) == 0:
                return ParameterGrid(to_grid_dict)
            else:
                final_result = []
                # TODO: Consider to retype the ParameterGrid structure and type
                all_possibilities = list(ParameterGrid(new_dict))
                for elem in all_possibilities:
                    to_grid_dict2 = to_grid_dict.copy()
                    for k, v in elem.items():
                        for k2, v2 in v.items():  # type: ignore
                            to_grid_dict2[k2] = [v2]
                    final_result.append(to_grid_dict2)
                return ParameterGrid(final_result)

        return ParameterGrid(to_grid_dict)

    @staticmethod
    def params_name_update(params: Paramset, name: str) -> Paramset:
        """
        Update keys adding step name prefix.

        :param params: dictionary with params
        :param name: step name

        :return: the updated paramset
        """
        return dict([(name + "__" + k, v) for k, v in params.items()])

    def iteration_count(self, name: str) -> int:
        """
        Count parameter combinations in one step.

        :param name: step name
        :return: int
        """
        return len(self._generate_grid(self.get_params_subset(name, True)))

    def log_time(self, iteration: int, len_iteration: int, n_folds: int) -> None:
        """
        Log times.

        :param iteration: iteration
        :param len_iteration: length of the iteration
        :param n_folds: number of folds
        """
        sys.stdout.flush()

        previous_time = self.actual_time
        self.actual_time = time.time()

        # self.tracker.print_diff()
        self.logger.info("======================================")

        if iteration == 0:
            duration_time = (
                (self.actual_time - previous_time) * (len_iteration - 1) * n_folds
            )
            self.logger.info("Expected duration: %f seconds" % duration_time)
            self.logger.info(
                "Expected end time: %s"
                % (
                    time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(self.start_time + duration_time),
                    )
                )
            )
        else:
            duration_time = (
                (self.actual_time - previous_time)
                * (len_iteration - iteration - 1)
                * n_folds
            )
            self.logger.info("Expected remaining duration: %f seconds" % duration_time)
            self.logger.info(
                "Expected end time: %s"
                % (
                    time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(self.start_time + duration_time),
                    )
                )
            )

        self.logger.info("======================================")
        sys.stdout.flush()

    def remapOutput(
        self, o: OptimizerOutput, params: Paramset, ith: int, name: str
    ) -> OptimizerOutput:
        """
        Calculate new hash with params used in the outer step of recursion; history and predictions remapping.

        :param o: OptimizerOutput
        :param params: param set
        :param ith: split number
        :param name: stage name
        :return: OptimizerOutput
        """
        self.logger.info("Fold: %d " % ith)

        newParams = (
            {
                k: union(self.params_name_update(params, name), v["params"])
                for k, v in o.history.items()
            }
            if o.history is not None
            else {}
        )
        mapping = {k: self._get_param_hash(v) for k, v in newParams.items()}
        p = (
            {
                mapping[k]: pd.concat({ith: v.loc[0, :]})
                for k, v in o.predictions.items()
            }
            if o.predictions is not None
            else {}
        )
        h = union(
            {mapping[k]: {"params": v} for k, v in newParams.items()},
            {mapping[k]: v for k, v in o.history.items()}
            if o.history is not None
            else {},
        )

        return OptimizerOutput(
            model=o.model,
            dataset=o.dataset,
            pred=pd.DataFrame(),
            history=h,
            predictions=p,
        )

    def loopOverParams(
        self, name: str, stage: Transformer, params_grid: ParameterGrid
    ) -> Iterator[Tuple[Paramset, Transformer, ConditionalParamspace]]:
        """
        Return a generator cycling on params combinations.

        :param name: stage name
        :param stage: stage model
        :param params_grid: params grid
        :yield: param combination, stage model, params of deeper step
        """
        for elem in params_grid:
            stage_clone = stage.clone()
            stage_clone.set_params(**elem)
            yield elem, stage_clone, self.get_params_subset(name, diff=True, elem=elem)

    def generateOutputs(
        self, dataset: PandasDataset, stage: Transformer, params: Paramset
    ) -> Iterator[OptimizerOutput]:
        """
        Return a generator cycling on split of train and valid.

        :param dataset: dataset
        :param stage: stage model
        :param params: params
        :yield: OptimizerOutput
        """
        for train, valid in self._split_strategy.split(dataset):

            trans = stage.train(train) if isinstance(stage, Estimator) else stage

            if isinstance(stage, Resampler):
                tm = cast(Resampler, trans).resample(dataset)
            else:
                tm = cast(Transformer, trans).transform(dataset)

            tail = PipelineEstimator(self._estimator.get_params()["steps"][1:])

            yield RecursiveOptimizer(
                tail,
                params,
                self._evaluator,
                split_strategy=IndexSplitter(
                    # TODO: Make resampler and transformer generic
                    train.index.intersection(cast(PandasDataset, tm).index), valid.index
                ),
                checkpoints=0,
                iteration_info=(self.counter, self.iteration[1]),
            ).set_store_predictions(True).optimize(tm)

    def optimize(self, dataset: PandasDataset) -> OptimizerOutput:
        """
        Get performances with recursion for the first steps in a Pipeline.

        :param dataset: dataset

        :raises TypeError: if dataset is not a PandasDataset

        :return: OptimizerOutput
        """
        if not isinstance(dataset, PandasDataset):
            raise TypeError("Input dataset must be a PandasDataset")

        n_folds = self._split_strategy.nSplits(dataset)

        self.iteration = (
            (0, len(self._generate_grid(self._parameter_space)) * n_folds)
            if self.iteration_info is None
            else self.iteration_info
        )
        self.counter = 0

        if (
            isinstance(self._estimator, PipelineEstimator)
            and len(self._estimator.get_params()["steps"]) > 1
        ):
            name, stage = self._estimator.get_params()["steps"][0]
            parameters_space = self.get_params_subset(name)
            params_grid = self._generate_grid(parameters_space)

            if self._checkpoints != 0:
                self.actual_time = time.time()

            it = (
                self.remapOutput(o, elem, ith, name)
                for elem, stage_clone, params in self.loopOverParams(
                    name, stage, params_grid
                )
                for ith, o in enumerate(
                    self.generateOutputs(dataset, stage_clone, params)
                )
            )

            if self._checkpoints != 0:
                count_checkpoint = ceil(
                    self.iteration_count(name) * n_folds / self._checkpoints
                )

                output = OptimizerOutput.mergeOutputs(
                    *[
                        self.checkpoint(
                            output, ibatch, dataset, n_folds, count_checkpoint
                        )
                        for ibatch, batch in enumerate(
                            groupIterable(
                                it,
                                self._checkpoints
                                * self.iteration_count(name)
                                * n_folds,
                            )
                        )
                        for output in OptimizerOutput.mergeOutputs(*batch)
                    ]
                )
            else:
                output = OptimizerOutput.mergeOutputs(*list(it))

            return self.finalizeOutput(next(output), dataset=dataset)
        else:
            return self.run_last_step(dataset)

    def checkpoint(
        self,
        output: OptimizerOutput,
        ibatch: int,
        dataset: PandasDataset,
        n_folds: int,
        count_checkpoint: Union[int, Iterable],
    ) -> OptimizerOutput:
        """
        Checkpoint based on batch size.

        :param output: OptimizerOutput
        :param ibatch: batch number
        :param dataset: datset
        :param n_folds: n folds
        :param count_checkpoint: number o groupIterable
        :return: OptimizerOutput
        """
        self._history = OptimizerOutput._mergeHistory(recompute=True)(
            self._history, output.history
        )
        # TODO: refactor is needed
        self._predictions = OptimizerOutput._mergePredictions(
            self._predictions, output.predictions
        )

        if (
            ibatch == count_checkpoint
            and len(self._generate_grid(self._parameter_space)) % self._checkpoints != 0
        ):
            return output

        self.log_time(
            len(self._history),
            len(self._generate_grid(self._parameter_space)),
            n_folds,
        )
        self.save_checkpoint(dataset, len(self._history))
        return output

    def finalizeOutput(
        self, output: OptimizerOutput, dataset: PandasDataset
    ) -> OptimizerOutput:
        """
        Perform final output aggregation.

        :param output: OptimizerOutput
        :param dataset: dataset
        :return: OptimizerOutput
        """
        if output.history is not None:
            self._history = output.history
        if output.predictions is not None:
            self._predictions = output.predictions
        return OptimizerOutput(
            model=self.best_model(dataset),
            dataset=dataset,
            pred=self.best_predictions,
            history=output.history,
            predictions=output.predictions,
        )

    def run_last_step(self, dataset: PandasDataset) -> OptimizerOutput:
        """
        Get performances of last step in a Pipeline or a model.

        :param dataset: dataset
        :return: OptimizerOutput
        """
        if len(self._parameter_space) == 0:
            _ = self.step(params={}, dataset=dataset)
        else:
            _params_grid = self._generate_grid(self._parameter_space)
            for iteration, paramset in enumerate(_params_grid):

                self.logger.info(
                    "Chunk: %d / %d"
                    % (self.iteration[0] + iteration, self.iteration[1])
                )

                if self._check_params_validity(paramset):
                    _ = self.step(params=paramset, dataset=dataset)
        return OptimizerOutput(
            model=self.best_model(dataset),
            dataset=dataset,
            pred=self.best_predictions,
            history=self._history,
            predictions=self._predictions if (self.store_predictions is True) else None,
        )
