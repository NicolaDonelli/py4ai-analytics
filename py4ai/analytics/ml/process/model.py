"""Modelling process."""
import os
import subprocess
import sys
import time
from typing import List, Optional, Union

import pandas as pd
from py4ai.data.model.core import Range
from py4ai.data.model.ml import PandasDataset
from py4ai.core.logging import WithLogging
from py4ai.core.utils.fs import create_dir_if_not_exists

from py4ai.analytics.ml.core import FeatureProcessing, TimeSeriesFeatureProcessing
from py4ai.analytics.ml.core.optimizer import CheckpointedOptimizer


# TODO [ND] this class should be reworked since there is no more a tester class and some methods could be updated
class ModelLab(WithLogging):
    """
    Class to manage the training, selection, validation and testing processes.

    This class allows prototipation of modelling pipelines with evaluations of performances.
    """

    def __init__(
        self, proc: FeatureProcessing, optimizer: CheckpointedOptimizer, tester
    ) -> None:
        """
        Initialize the class.

        :param proc: feature processing class
        :param optimizer: optimizer class
        :param tester: tester class
        """
        self.logger.info(" ")
        self.logger.info(" PID: %s" % str(os.getpid()))
        self.logger.info(" ")

        # self.predict_type = predict_type
        self.proc = proc
        self.optimizer = optimizer
        self.tester = tester

        self._run_name: Optional[str] = None

    def set_name(self, name: str) -> "ModelLab":
        """
        Set the name to be used as folder name when saving.

        :param name: name to be used when saving
        :return: self
        """
        self._run_name = name
        return self

    def set_path(self, dir_name: str) -> "ModelLab":
        """
        Set directory where models are stored and create it if not existing.

        :param dir_name: name of the target directory
        :return: self
        :raises OSError: if path does not exist
        """
        self._dir_name = dir_name

        if os.path.exists(self._dir_name):
            if not os.path.isdir(self._dir_name):
                raise OSError("Path {} is not a directory".format(self._dir_name))
            else:
                pass
        else:
            create_dir_if_not_exists(self._dir_name)

        return self

    def write_predictions(
        self,
        predictions: pd.Series,
        train: Optional[PandasDataset] = None,
        test: Optional[PandasDataset] = None,
    ) -> None:
        """
        Write predictions in the defined path, with the give name.

        :param predictions: predictions series
        :param train: train features and labels
        :param test: test features and labels
        """
        folder_name = self.logResult(lambda x: "  Writing outputs to {}".format(x))(
            os.path.join(self._dir_name, self.run_name)
        )

        subprocess.check_output(("mkdir -p {}".format(folder_name)).split())

        if train is not None:
            train_file = self.logResult(" Writing out training set")(
                os.path.join(folder_name, "train")
            )
            train.write(train_file)
        if test is not None:
            test_file = self.logResult(" Writing out test set")(
                os.path.join(folder_name, "test")
            )
            test.write(test_file)

        prediction_file = self.logResult(
            lambda x: " Writing out predictions on test set to {}".format(x)
        )(os.path.join(folder_name, "prediction"))

        if isinstance(predictions, pd.DataFrame):
            predictions.to_pickle(prediction_file)
        else:
            import pickle as pk

            with open(prediction_file, "wb") as fout:
                pk.dump(predictions, fout)

    def _read_last_checkpoint(self, folder_name: str) -> str:
        """
        Get most recent checkpoint from its name, provided a meaningful syntax.

        :param folder_name: name of the checkpoint folder
        :return: checkpoint
        """
        import glob
        import re

        checkpoint = self.logResult(
            lambda x: " Last valid checkpoint: {}".format(x)
        )(
            sorted(
                [
                    re.sub(".*/", "", x)
                    for x in glob.glob(os.path.join(folder_name, ".checkpoints/*"))
                ],
                reverse=True,
            )[0]
        )

        return checkpoint

    def from_checkpoint(self, checkpoint: Optional[str] = None) -> None:
        """
        Resume from checkpoint.

        :param checkpoint: checkpoint to resume
        :raises ValueError: if directory name is not set
        """
        import pickle

        if self._dir_name is None:
            raise ValueError(
                "dir_name set to none. Must be provided when running from checkpoints"
            )

        folder_name = os.path.join(self._dir_name, self.run_name)

        self.logger.debug(" Testing from checkpoint {}".format(checkpoint))
        self.logger.debug(
            " Using training set and test set in folder {}".format(folder_name)
        )

        with open(folder_name + "/train", "rb") as fid:
            train = pickle.load(fid)
        with open(folder_name + "/test", "rb") as fid:
            test = pickle.load(fid)

        self.proc = FeatureProcessing.load(
            os.path.join(os.path.join(folder_name, "proc.p"))
        )

        try:
            if checkpoint is None:
                checkpoint = self._read_last_checkpoint(folder_name)
            self.optimizer.resume_from_checkpoint(
                os.path.join(folder_name, ".checkpoints", checkpoint)
            )
        except (IOError, IndexError):
            self.logger.error(" No checkpoint found. Starting new optimizer.")

        self.logger.info(" ")
        self.logger.info("  --> Configuration optimization...")
        sys.stdout.flush()
        res = self.optimizer.set_checkpoints_path(
            os.path.join(folder_name, ".checkpoints")
        ).optimize(train)

        self.logger.info(" ")
        self.logger.info("      Writing out optimization outputs ")
        sys.stdout.flush()
        _ = self.logResult(lambda x: "      Writing model to {}".format(x))(
            res.write(path=self._dir_name, model_name=self.run_name, overwrite=True)
        )

        res.model.save(os.path.join(folder_name, "model.p"))

        self.logger.info("  --> Testing best model ")
        self.logger.info(" ")
        sys.stdout.flush()
        prediction = self.tester.run(res.model, train, test)

        prediction_path = self.logResult(
            lambda x: "      Writing out prediction on test set to {}".format(x)
        )(os.path.join(folder_name, "test.p"))

        sys.stdout.flush()
        prediction.to_pickle(prediction_path)
        self.logger.info(" ")

    @property
    def run_name(self) -> str:
        """
        Property returning the name (folder name).

        If the name is not set, the time (format="%Y%m%d_%H:%M:%S") will be returned.

        :return: run name
        """
        if self._run_name is None:
            self._run_name = time.strftime("%Y%m%d_%H:%M:%S")
        return self._run_name

    def execute(
        self,
        train_range: Union[Range, List],
        test_range: Union[Range, List],
        params_path: Optional[str] = None,
    ) -> None:
        """
        Execute the modelling pipeline, compute predictions on the test set and save results.

        :param train_range: range for the training set (Range object if we are modelling a time series) a List otherwise
        :param test_range: range for the test set (Range object if we are modelling a time series) a List otherwise
        :param params_path: path to params.py file
        """
        start_time = time.time()
        folder_name = create_dir_if_not_exists(
            os.path.join(self._dir_name, self.run_name)
        )

        # Comment this out if you are not working on a git repo:
        try:
            with open(os.path.join(folder_name, "code_version.txt"), "w") as fid:
                fid.write(
                    "\n".join(
                        subprocess.check_output(["git", "log"])
                        .decode("UTF-8")
                        .split("\n")[:6]
                    )
                )
        except subprocess.CalledProcessError:
            pass

        if params_path is not None:
            subprocess.check_output(
                f"cp {os.path.join(params_path, 'params.py')} {folder_name}", shell=True
            )

        self.logger.info(" ")
        self.logger.info("  --> Features space creation for training and testing...")
        sys.stdout.flush()

        train, test = (
            self.proc.split_by_time_range(train_range, test_range)
            if isinstance(train_range, Range)
            and isinstance(test_range, Range)
            and isinstance(self.proc, TimeSeriesFeatureProcessing)
            else self.proc.split_by_indices(train_range, test_range)
        )

        self.logger.info(
            "       Writing out training and test set in folder {}".format(folder_name)
        )
        train.write(os.path.join(folder_name, "train"))
        test.write(os.path.join(folder_name, "test"))

        self.logger.info(" ")
        self.logger.info("  --> Configuration optimization...")
        sys.stdout.flush()
        res = self.optimizer.set_checkpoints_path(
            os.path.join(folder_name, ".checkpoints")
        ).optimize(train)

        self.logger.info(" ")
        self.logger.info("      Writing out optimization outputs ")
        sys.stdout.flush()
        model_name = res.write(
            path=self._dir_name, model_name=self.run_name, overwrite=True
        )
        self.logger.info("      Model writing to {}".format(model_name))

        self.proc.write(os.path.join(folder_name, "proc.p"))
        res.model.write(os.path.join(folder_name, "model.p"))

        self.logger.info(" ")
        self.logger.info("  --> Testing best model ")
        sys.stdout.flush()
        prediction = self.tester.run(res.model.estimator, train, test)

        self.logger.info("Best params: %s" % str(res.model.estimator.get_params()))

        test_path = os.path.join(model_name, "test.p")
        self.logger.info(
            "      Writing out prediction on test set to {}".format(test_path)
        )
        sys.stdout.flush()
        prediction.to_pickle(test_path)
        self.logger.info(" ")
        self.logger.info(
            "---- Elapsed time: %s seconds ----" % str(time.time() - start_time)
        )
