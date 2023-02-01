"""Implementation of the Discretizer class and related helper classes."""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from py4ai.data.model.ml import PandasDataset
from typing_extensions import TypedDict

from py4ai.analytics.ml.core import Numeric, Transformer


class ToDiscretizeElement(TypedDict):
    """Represent the threholds and label names of a field to be discretized."""

    threshold: List[Numeric]
    label_names: List[str]


class ToDiscretize(TypedDict):
    """Represent the list of features and labels to be discretized."""

    features: Dict[str, ToDiscretizeElement]
    labels: Dict[str, ToDiscretizeElement]


def toMultiLevels(
    y: pd.Series, thres: List[Numeric], label_names: Optional[List[str]] = None
) -> pd.Series:
    """
    Apply getClass to a series or dataframe, gives the class a value belongs given a list of thresholds.

    :param y: Series or dataframe to discretize
    :param thres: list of thresholds
    :param label_names: list of labels for the discretization

    :return: Discretized series
    """
    return y.apply(lambda x: getClass(x, thres, label_names))


def getClass(
    x: float, thres: List[Numeric], label_names: Optional[List[str]] = None
) -> Union[str, int]:
    """
    Map the given x value in the belonging class defined by a the list of thresholds thres.

    :param x: value to map
    :param thres: list of thresholds
    :param label_names: list of labels for the discretization

    :return: class where x belongs
    """
    v = 0
    thresValue = thres[v]
    while x > thresValue:
        v += 1
        try:
            thresValue = thres[v]
        except IndexError:
            v = len(thres)
            break
    return label_names[v] if label_names is not None else v


class Discretizer(Transformer):
    """Discretize a series with threshold and labels."""

    def __init__(self, to_discretize: ToDiscretize) -> None:
        """
        Class instance initializer.

        :param to_discretize: dict with "features" and "labels" as possible keys.
            Values are dictionaries with columns names as keys and a dictionary as value with
            "threshold" and "label_names" keys.

            threshold: float or list of threshold
            label_names: label names that will be assigned after the discretization. If labels are not specified, they
            will be assigned from 0. They must be one fewer than the number of specified thresholds.

        EX:
        {"features": {"f_1": {"threshold": [-1, 1], "label_names": [0, 1, 2]}, "f_2": {"threshold": [-1, 1]}},
        "labels": None}
        """
        self.check_discretize_dict(to_discretize)
        self.to_discretize = to_discretize

    def check_discretize_dict(self, to_discretize: ToDiscretize) -> None:
        """Check that the structure of the ToDiscretize object is valid.

        :param to_discretize: instance of ToDiscretize to be checked
        """
        if not isinstance(to_discretize, dict):
            self.logger.error(
                f"A dictionary is expected, {type(to_discretize)} is passed to to_dicretize"
            )

        for labels_features, col_dict in to_discretize.items():
            if labels_features in ["labels", "features"] and isinstance(col_dict, dict):
                for thr_lab_dict in col_dict.values():
                    if "threshold" not in thr_lab_dict:
                        self.logger.error(
                            "Each features must have a specified threshold"
                        )
                    elif (
                        "label_names" in thr_lab_dict
                        and thr_lab_dict["label_names"] is not None
                    ):
                        self.check_label_names(
                            thr_lab_dict["threshold"], thr_lab_dict["label_names"]
                        )
            else:
                self.logger.error(
                    "to_dicretize can only have 'features' and 'labels' as keys and a dict as values"
                )

    def check_label_names(self, threshold: List[Numeric], label: List[str]) -> None:
        """
        Check constraint on label and threshold shape.

        :param threshold: parameter thresholds
        :param label: label names
        """
        if len(threshold) + 1 != len(label):
            self.logger.error("Labels must be one fewer than the number of thresholds")

    @staticmethod
    def check_threshold(th: Union[Numeric, List[Numeric]]) -> List[Numeric]:
        """
        Check threshold shape.

        :param th: threshold(s)

        :raises TypeError: if th is not numeric neither list

        :return: th if list, [th] if numeric
        """
        if isinstance(th, float) or isinstance(th, int):
            return [th]
        elif isinstance(th, list):
            return th
        else:
            raise TypeError("Incorrect threshold passed. It must be a list or a number")

    @staticmethod
    def discr(
        series: pd.Series, threshold: List[Numeric], label_names: List[str] = None
    ) -> pd.Series:
        """
        Discretize a series.

        :param series: series
        :param threshold: thresholds
        :param label_names: label names

        :return: series
        """
        if label_names is None:
            label_names = list(np.arange(len(threshold) + 1))
        if isinstance(series, pd.DataFrame):
            series = series.squeeze()
        return toMultiLevels(series, threshold, label_names)

    def apply(self, dataset: PandasDataset) -> PandasDataset:
        """
        Discretize a given subset of columns.

        :param dataset: dataset

        :raises TypeError: if not PandasDataset

        :return: discretized dataset
        """
        if not isinstance(dataset, PandasDataset):
            raise TypeError("dataset must be of type PandasDataset")

        features = dataset.features.copy()
        labels = dataset.labels.copy() if dataset.labels is not None else None

        if "features" in self.to_discretize:
            for col, v in self.to_discretize["features"].items():
                features.loc[:, col] = (
                    self.discr(features.loc[:, col], v["threshold"], v["label_names"])
                    if "label_names" in v
                    else self.discr(features.loc[:, col], v["threshold"])
                )

        if "labels" in self.to_discretize:
            for col, v in self.to_discretize["labels"].items():
                labels.loc[:, col] = (
                    self.discr(labels.loc[:, col], v["threshold"], v["label_names"])
                    if "label_names" in v
                    else self.discr(labels.loc[:, col], v["threshold"])
                )

        return PandasDataset(features, labels)
