"""Implementation of abstact classes and utils related to the Optimizer."""
import hashlib
import os
import re
import sys
import time
from abc import ABC
from datetime import datetime
from functools import reduce
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, TypeVar, Generic, cast

import dill
import numpy as np
import pandas as pd
from py4ai.data.model.ml import TDatasetUtilsMixin, PandasDataset
from py4ai.core.logging import WithLogging
from py4ai.core.utils.decorators import lazyproperty as lazy
from py4ai.core.utils.dict import filterNones, union
from py4ai.core.utils.fs import mkdir
from typing_extensions import TypedDict

from py4ai.analytics.ml import PathLike
from py4ai.analytics.ml.core import (
    Estimator,
    Evaluator,
    Metric,
    Optimizer,
    Splitter,
    Transformer,
    DATASET
)
from py4ai.analytics.ml.process.saver import _saveAs

Paramset = Dict[str, Any]


class HistoryRow(TypedDict):
    """Dictionary representing a row in optimizer history."""

    date: str
    params: Dict[str, Any]
    train_mean_metric: float
    valid_mean_metric: float
    train_metrics: List[Metric]
    valid_metrics: List[Metric]


class HistoryRowPred(HistoryRow):
    """History row with predictions."""

    predictions: pd.DataFrame


History = Dict[str, HistoryRow]
Predictions = Dict[str, Union[pd.Series, pd.DataFrame]]


def _default_error_combiner(x: HistoryRow) -> float:
    """
    Get validation performance from the history of a optimizer.

    :param x: history from a optimizer

    :return: validation mean error
    """
    return x["valid_mean_metric"]


def _get_param_hash(params):
    """
    Get the hash given a set of params.

    :param params: set of parameters
    :return: hash
    """
    return hashlib.sha256((dill.dumps(params))).hexdigest()[:12]


def recomputePerformance(history_row):
    """
    Recompute performances: mean of the corresponding lists values.

    :param history_row: history

    :return: history with updated performances
    """
    train_metrics = history_row["train_metrics"]
    valid_metrics = history_row["valid_metrics"]

    aggregated = {
        "train_mean_metric": np.array(
            [metric.value for metric in train_metrics]
        ).mean(),
        "valid_mean_metric": np.array(
            [metric.value for metric in valid_metrics]
        ).mean(),
    }

    return union(history_row, aggregated)


def mergeRuns(*rs: HistoryRow) -> HistoryRow:
    """
    Merge different runs related to the same history.

    :param rs: list of runs (dictionaries)
    :return: dictionary
    """
    # Check that all params pertain to the same run
    assert len(set([_get_param_hash(r["params"]) for r in rs])) == 1

    return HistoryRow(
        date=rs[0]['date'],
        params=rs[0]['params'],
        # train_mean_metric=np.array([metric for r in rs for metric in r["train_metrics"]]).mean(),
        # valid_mean_metric=np.array([metric for r in rs for metric in r["valid_metrics"]]).mean(),
        train_mean_metric=np.NaN,
        valid_mean_metric=np.NaN,
        # TODO this could be done as sum([r['train_metrics'] for r in rs], [])
        train_metrics=[metric for r in rs for metric in r["train_metrics"]],
        valid_metrics=[metric for r in rs for metric in r["valid_metrics"]]
    )


class BasicOptimizer(Optimizer[DATASET], WithLogging, ABC):
    """Base class for optimizers."""

    @staticmethod
    def _error_combiner(x):
        return _default_error_combiner(x)

    def __init__(
        self, estimator: Estimator, evaluator: Evaluator, split_strategy: Splitter
    ):
        """
        Class instance initializer.

        :param estimator: transformer with fit and transform (accepting Dataframes and Series)
        :param evaluator: evaluator
        :param split_strategy: split strategy, in a similar fashion as done for kfold sklearn classes
        """
        self._estimator = estimator
        self._split_strategy = split_strategy
        self._evaluator = evaluator

        self._history: History = {}
        self._predictions: Predictions = {}
        self.best_run: Optional[str] = None
        self._store_predictions: bool = False
        self._run_id: Optional[str] = None

    @property
    def store_predictions(self) -> bool:
        """
        Get store predictions option.

        :return: store predictions option
        """
        return self._store_predictions

    @store_predictions.setter
    def store_predictions(self, value: bool) -> None:
        """
        Set store prediction option.

        :param value: new value
        """
        self._store_predictions = value

    def set_store_predictions(self, value: bool) -> "BasicOptimizer":
        """
        Set store prediction option.

        :param value: new value
        :return: self
        """
        self.store_predictions = value
        return self

    @staticmethod
    def _get_param_hash(params: Dict[str, Any]) -> str:
        """
        Get the hash given a set of params.

        :param params: set of parameters
        :return: hash
        """
        return _get_param_hash(params)

    @staticmethod
    def _sort_history(
        history: History, error_combiner: Callable[[HistoryRow], float]
    ) -> List[Tuple[str, HistoryRow]]:
        """
        Sort parameters and get the best combination.

        :param history: history dictionary
        :param error_combiner: function to use for sorting

        :return: sorted history
        """
        return [
            (k, history[k])
            for k in sorted(
                history, key=lambda x: error_combiner(history[x]), reverse=True
            )
        ]

    def get_predictions(
        self, key: Optional[str] = None
    ) -> Union[Predictions, Union[pd.Series, pd.DataFrame]]:
        """
        Get predictions, either all or by key.

        :param key: predictions key
        :return: target prediction
        """
        if key is None:
            return self._predictions
        else:
            return self._predictions[key]

    def _get_best(self) -> str:
        """
        Sort parameters and get the best combination.

        :return: best parameters hash
        """
        return self._sort_history(self._history, self._error_combiner)[0][0]

    @property
    def best_predictions(self) -> pd.DataFrame:
        """
        Get the value of best performances so far.

        :return: best performance value
        """
        return self._predictions[self._get_best()]

    @property
    def best_params(self) -> Dict[str, Any]:
        """
        Get the paramaters with best performance so far.

        :return: best parameters
        :rtype: dict
        """
        return self._history[self._get_best()]["params"]

    @property
    def best_estimator(self) -> Estimator:
        """
        Get best estimator.

        :return: best estimator instance
        """
        estimator = self._estimator.clone()
        estimator.set_params(**self.best_params)
        return estimator

    def best_model(self, dataset: TDatasetUtilsMixin) -> Transformer:
        """
        Refit the pipepline to the best parameters.

        :param dataset: input data
        :return: trained best model
        """
        return self.best_estimator.train(dataset)

    def _evaluate(
        self, estimator: Estimator, params: Dict[str, Any], dataset: TDatasetUtilsMixin
    ) -> HistoryRowPred:
        """
        Evaluate a given estimator (idenfified by the parameters) over cross-validation splits.

        :param estimator: Estimator
        :param params: Hyper-parameters to be evaluated
        :param dataset: input data

        :return: dictionary with the results of the run
        """
        estimator.set_params(**params)

        train_metrics = []
        valid_metrics = []

        predictions = {}

        for i_fold, (train, valid) in enumerate(self._split_strategy.split(dataset)):
            self.logger.debug(
                "Features: %s" % ",".join(list(map(str, train.getFeaturesAs('pandas').index)))
            )
            self.logger.debug(
                "Labels: %s" % ",".join(list(map(str, train.getLabelsAs('pandas').index)))
            )

            fitted = estimator.train(train)

            train_pred = fitted.transform(train)
            valid_pred = fitted.transform(valid)

            train_metrics.append(self._evaluator.evaluate(train_pred))
            valid_metrics.append(self._evaluator.evaluate(valid_pred))

            predictions[i_fold] = valid_pred.getFeaturesAs("pandas")

        return HistoryRowPred(
            date=datetime.today().isoformat(),
            params=params,
            train_mean_metric=np.array(
                [metric.value for metric in train_metrics]
            ).mean(),
            valid_mean_metric=np.array(
                [metric.value for metric in valid_metrics]
            ).mean(),
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            predictions=pd.concat(predictions),
        )

    def step(self, dataset: TDatasetUtilsMixin, params: Dict[str, Any]) -> HistoryRow:
        """
        Run the optimization process.

        :param params: dictionary with estimator parameters
        :param dataset: input dataset

        :return: history row
        """
        history_row = self._evaluate(self._estimator.clone(), params, dataset)
        # TODO: refactor the getting of predictions in a better way: no mutability
        predictions = history_row["predictions"]

        key = self._get_param_hash(params)
        self._history[key] = HistoryRow(
            date=history_row['date'],
            params=history_row['params'],
            train_metrics=history_row['train_metrics'],
            valid_metrics=history_row['valid_metrics'],
            train_mean_metric=history_row['train_mean_metric'],
            valid_mean_metric=history_row['valid_mean_metric']
        )
        if self.store_predictions:
            self._predictions[key] = predictions
        elif self._get_best() == key:
            self._predictions = {key: predictions}

        return history_row

    def _check_params_validity(self, params: Dict[str, Any]) -> bool:
        """
        Verify if parameters where used in a previous iteration.

        :param params: dictionary with parameters
        :return: check's boolean output
        """
        if params:
            if self._get_param_hash(params) in self._history.keys():
                self.logger.debug("Parameters already present in the history")
                return False
            else:
                return True
        else:
            return False

    def get_report(
        self, sort: bool = True, limit: int = 5, extended: bool = False
    ) -> pd.DataFrame:
        """
        Create optimization results report.

        :param sort: Determines if the report should be sorted by the validation performance metric. Default = True
        :param limit: Number of rows to show
        :param extended: Determines if the performance metrics for each fold of each iteration are shown

        :return: report
        """
        if sort is False:
            report = pd.DataFrame(self._history).T
        else:
            history = self._sort_history(self._history, self._error_combiner)
            report = pd.DataFrame(
                [x[1] for x in history], index=[x[0] for x in history]
            )

        columns = ["params", "valid_mean_metric", "train_mean_metric"] + (
            ["train_metrics", "valid_metrics"] if extended else []
        )

        return (report[columns] if limit < 0 else report.head(limit))[columns]


class OptimizerOutput(WithLogging):
    """Store and save optimization run information."""

    def __init__(
        self,
        model: Transformer,
        dataset: PandasDataset,
        pred: pd.DataFrame,
        history: Optional[History],
        predictions: Optional[Predictions] = None,
    ) -> None:
        """
        Class instance initializer.

        :param model: optimized model
        :param dataset: training set
        :param pred: predictions on validation set obtained with optimized model
        :param history: history dictionary from Optimizer
        :param predictions: predictions dictionary from Optimizer
        """
        self.model = model
        self.dataset = dataset
        self.pred = pred

        self.history = history
        self.predictions = predictions

    def predictions_to_df(self) -> pd.DataFrame:
        """
        Cast predictions to dataframe.

        :return: dataframe containing the predictions
        """
        return pd.concat(self.predictions, axis=1)

    @staticmethod
    def _getIndexLevels(df: pd.DataFrame) -> int:
        """
        Return index levels.

        :param df: dataframe
        :return: number of index level
        """
        try:
            foldingNumber = len(df.index.levels)
        except AttributeError:
            foldingNumber = 1
        return foldingNumber

    @lazy
    def _features(self) -> pd.DataFrame:
        return self.dataset.getFeaturesAs("pandas")

    @lazy
    def _labels(self) -> pd.DataFrame:
        return self.dataset.getLabelsAs("pandas")

    @lazy
    def has_folding_number(self) -> bool:
        """
        Check if there is a folding number.

        :return: boolean indicating the presence of a folding
        """
        return self._getIndexLevels(self._labels) < self._getIndexLevels(self.pred)

    def _addFoldingNumber(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add folding number.

        :param df: input dataframe
        :return: dataframe with folds
        :raises ValueError: if this instance does not have a folding number
        """
        if self.has_folding_number is False:
            raise ValueError(
                "Cannot add folding number since OptimizerOutput pred do not have such information. "
                "It is possible that optimization did not involved a folding strategy."
            )
        return pd.concat(
            {i: df.loc[k.loc[i].index] for i, k in self.pred.groupby(level=0)}
        )

    def trainingPredictions(self) -> PandasDataset:
        """
        Compute get predictions on train.

        :return: dataset with predictions on train as features and true value as labels
        """
        return self.dataset.createObject(
            self.pred,
            self.dataset.getLabelsAs('pandas')
            if self.has_folding_number is False
            else self._addFoldingNumber(self._labels),
        )

    def write(
        self,
        path: PathLike,
        model_name: Optional[str] = None,
        overwrite: bool = False,
        suffix: Optional[str] = None,
    ) -> PathLike:
        """
        Save to file model, report, train and test datasets, and train ans test predictions.

        :param path: path to save files
        :param model_name: name for the model, if None it uses the current time in timestamp format. Default = None.
        :param overwrite: if true overwrites if it finds a model with the same name
        :param suffix: suffix to add to the model name. Default = None.

        :return: path to model, train and test datasets, and train and test predictions saved in pickle format
        """
        if not os.path.isdir(path):
            mkdir(path)

        if model_name is None:
            modelClass = re.sub(".*\\.", "", str(self.model.__class__))[:-2]
            timestamp = time.strftime("%Y%m%d_%H:%M:%S_")
            suffix = "_" + suffix if suffix is not None else ""
            model_name = timestamp + modelClass + suffix

        output = filterNones(
            {
                "model": self.model,
                "_train": self.dataset,
                "train.p": self.pred,
                "history": self.history if self.history is not None else None,
                "predictions": self.predictions
                if self.predictions is not None
                else None,
            }
        )

        self.logger.info("Saving OptimizerOutput with %s" % ",".join(output.keys()))

        return _saveAs(output, os.path.join(path, str(model_name)), overwrite=overwrite)

    @staticmethod
    def read(folder_name: PathLike, mode: str = "rb") -> "OptimizerOutput":
        """
        Import a saved model.

        :param folder_name: folder where the files are located
        :param mode: open file mode. This parameter is here only for backward compatibility, it should not be changed.

        :return: OptimizerOutput with model info
        """
        inputs = ["model", "_train", "train.p", "history", "predictions"]
        data = {}
        for input in inputs:
            if os.path.isfile(os.path.join(folder_name, input)):
                with open(os.path.join(folder_name, input), mode) as fid:
                    data[input] = dill.load(fid)

        return OptimizerOutput(
            model=data["model"],
            dataset=data["_train"],
            pred=data["train.p"],
            history=data.get("history", None),
            predictions=data.get("predictions", None),
        )

    @staticmethod
    def mergeOutputs(*os: "OptimizerOutput") -> Iterator["OptimizerOutput"]:
        """
        Merge outputs.

        :param os: list of OptimizerOutput from different iterations
        :yield: final OptimizerOutput
        """
        if any([o.predictions is None for o in os]):
            p = None
        else:
            p = OptimizerOutput._mergePredictions(*[o.predictions for o in os])

        if any([o.history is not None for o in os]):
            h = OptimizerOutput._mergeHistory(recompute=True)(*[o.history for o in os])
        else:
            h = None

        yield OptimizerOutput(
            # TODO: refactor the entire class
            model=None, dataset=None, pred=np.nan, history=h, predictions=p  # type :  ignore
        )

    @staticmethod
    def _mergePredictions(*preds: Predictions) -> Predictions:
        """
        Merge predictions with concatenation.

        :param preds: list of predictions (dictionaries)
        :return: dictionary
        """
        allKeys = reduce(lambda agg, x: agg.union(x), preds[1:], set(preds[0]))
        return {k: pd.concat([p[k] for p in preds if k in p.keys()]) for k in allKeys}

    @staticmethod
    def _mergeHistory(recompute: bool = False) -> Callable[[History], History]:
        """
        Merge history.

        :param recompute: True or False, if metrics should be recomputed, default = False
        :return: a callable that merges computation history
        """

        def wrap(*hs: History) -> History:
            """
            Merge histories.

            :param hs: list of histories (dictionaries)
            :return: dictionary
            """
            f = recomputePerformance if (recompute is True) else lambda x, metric: x
            allKeys = reduce(lambda agg, x: agg.union(x), hs[1:], set(hs[0]))
            return {
                k: f(mergeRuns(*[h[k] for h in hs if k in h.keys()])) for k in allKeys  # type: ignore
            }
        return wrap


class CheckpointedOptimizer(BasicOptimizer[DATASET], ABC, Generic[DATASET]):
    """Restartable optimization process."""

    _checkpoints: int

    def __init__(
        self,
        estimator: Estimator,
        evaluator: Evaluator,
        split_strategy: Splitter,
        checkpoints_path: Optional[Union[str, "os.PathLike[str]"]],
        checkpoint_refit: bool,
        checkpoint_overwrite: bool,
    ) -> None:
        """
        Class instance initializer.

        :param estimator: transformer with fit and transform (accepting Dataframes and Series)
        :param evaluator: evaluator
        :param split_strategy: split strategy
        :param checkpoints_path: path to save checkpoints. If None no checkpoints are saved.
        :param checkpoint_refit: if True refits the estimator with the new best parameters after the checkpoint.
        :param checkpoint_overwrite: if True it overwrites the latest checkpoint
        """
        super(CheckpointedOptimizer, self).__init__(
            estimator=estimator, evaluator=evaluator, split_strategy=split_strategy
        )
        self._checkpoints_path = checkpoints_path
        self._checkpoint_refit = checkpoint_refit
        self._checkpoint_overwrite = checkpoint_overwrite

    def set_checkpoints(self, value: int) -> "CheckpointedOptimizer":
        """
        Set number of steps every which perform a checkpoint.

        :param value: number to set
        :return: self
        """
        self._checkpoints = value
        return self

    def set_checkpoints_path(self, value: PathLike) -> "CheckpointedOptimizer":
        """
        Set path to save checkpoints.

        :param value: path to save checkpoints
        :return: self
        """
        self._checkpoints_path = value
        return self

    def save_checkpoint(self, dataset: PandasDataset, iteration: int) -> None:
        """
        Save a checkpoint from the optimizer.

        :param dataset: training dataset
        :param iteration: iteration of the run
        """
        self.logger.info("=================")
        self.logger.info("Saving checkpoint")
        self.logger.info("=================")
        self.logger.info(" ")
        self.logger.info(f"Iterations: {iteration}")
        self.logger.info(f"Best params up to now: {self.best_params}")
        self.logger.info(f"Best run: {self._get_best()}")
        if (self.best_run != self._get_best()) & self._checkpoint_refit:
            self.logger.info("Best parameters has changed. Retraining new best model")
        self.best_run = self._get_best()
        hash = self._save_checkpoint(dataset)
        self.logger.info(f"Checkpoint: {hash}")
        sys.stdout.flush()

    def _save_checkpoint(self, dataset: PandasDataset) -> Optional[str]:
        """
        Actual saving a checkpoint from the optimizer.

        :param dataset: training dataset
        :return: name of the folder containing optimizer output
        """
        if self._checkpoints_path is not None:
            res = OptimizerOutput(
                model=self.best_model(dataset),
                dataset=dataset,
                history=self._history,
                pred=self.best_predictions,
                predictions=self._predictions
                if (self.store_predictions is True)
                else None,
            )

            name = (
                "99999999999999"
                if self._checkpoint_overwrite
                else datetime.utcnow().strftime("%Y%m%d%H%M%S")
            )
            res.write(path=self._checkpoints_path, model_name=name, overwrite=True)
            return name
        else:
            return None

    def resume_from_checkpoint(self, checkpoint: str) -> None:
        """
        Resume optimizer from a saved checkpoint.

        :param checkpoint: checkpoint folder name
        """
        if self._checkpoints_path is not None:
            with open(
                os.path.join(self._checkpoints_path, checkpoint, "model"), "rb"
            ) as fid:
                self._estimator = dill.load(fid).clone()

            with open(
                os.path.join(self._checkpoints_path, checkpoint, "history"), "rb"
            ) as fid:
                self._history = dill.load(fid)

            if os.path.isfile(
                os.path.join(self._checkpoints_path, checkpoint, "predictions")
            ):
                with open(
                    os.path.join(self._checkpoints_path, checkpoint, "predictions"),
                    "rb",
                ) as fid:
                    self._predictions = dill.load(fid)
