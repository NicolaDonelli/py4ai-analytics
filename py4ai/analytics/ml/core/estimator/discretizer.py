"""Implementation of the discretizer estimator classes."""
from typing import Dict, List, cast

import pandas as pd
from py4ai.data.model.ml import PandasDataset

from py4ai.analytics.ml.core import Estimator, Numeric
from py4ai.analytics.ml.core.enricher.transformer.discretizer import Discretizer, ToDiscretize, ToDiscretizeElement


class QuantileEstimator(Estimator):
    """Discretize features and labels based on its quantiles."""

    def __init__(self, to_discretize_q: ToDiscretize):
        """
        Class insitance initializer.

        :param to_discretize_q: dict with "features" and "labels" as possible keys.
            Values are dictionaries with columns names as keys and a dictionary as value with
            "q_list" and "label_names" keys.
            quantile: list of quantiles (float numbers between 0 and 1)
            label_names: label names that will be assigned after the discretization. If labels are not specified, they
            will be assigned from 0. They must be one fewer than the number of specified thresholds.
            EX:
            {"features": {"f_1": {"q_list": [0.3, 0.7], "label_names": [0, 1, 2]}, "f_2": {"q_list": [0.4, 0.6]}},
            "labels": None}
            threshold: float or list of threshold
        """
        self.check_quantile_dict(to_discretize_q)
        self.to_discretize_q = to_discretize_q

    def check_quantile_dict(self, quantile_list: ToDiscretize) -> None:
        """
        Check quantile_list param.

        :param quantile_list: quantile list
        """
        if not isinstance(quantile_list, dict):
            self.logger.error(
                "A dictionary is expected, %s is passed to to_dicretize"
                % type(quantile_list)
            )

        for labels_features, col_dict in quantile_list.items():
            if labels_features in ["labels", "features"] and isinstance(col_dict, dict):
                for q_lab_dict in col_dict.values():
                    if "q_list" not in q_lab_dict:
                        self.logger.error(
                            "Each features must have a specified quantile list"
                        )
                    else:
                        self.check_quantile(q_lab_dict["q_list"])
            else:
                self.logger.error(
                    "to_dicretize can only have 'features' and 'labels' as keys and a dict as values"
                )

    def check_quantile(self, q_list: List[float]) -> None:
        """
        Check quantiles are float numbers between 0 and 1.

        :param q_list: quantile list
        """
        for n in q_list:
            if n < 0 or n > 1:
                self.logger.error("Quantiles must be float between 0 and 1")

    @staticmethod
    def compute_threshold(series: pd.Series, q_list: List[float]) -> List[Numeric]:
        """
        Compute quantile threshold.

        :param series: data
        :param q_list: quantile list
        :return: list of thresholds
        """
        threshold_list = [series.quantile(q) for q in q_list]
        return sorted(threshold_list)

    def train(self, dataset: PandasDataset) -> Discretizer:
        """
        Create transformer able to discretize based on quantiles calculated from input data.

        :param dataset: Dataset to train discretization on
        :return: trained discretizer
        """
        quant_dict = ToDiscretize(features={}, labels={})

        features = dataset.features.copy()
        labels = dataset.labels.copy()

        for f_l, values_f_l in self.to_discretize_q.items():
            quant_dict_f_l = {}
            for col, values_col in cast(Dict[str, ToDiscretizeElement], values_f_l).items():
                series = (
                    features.loc[:, col] if f_l == "features" else labels.loc[:, col]
                )

                threshold = self.compute_threshold(series, values_col["q_list"])
                label_names = (
                    values_col["label_names"] if "label_names" in values_col else None
                )

                quant_dict_f_l[col] = {
                    "threshold": threshold,
                    "label_names": label_names,
                }

            quant_dict[f_l] = quant_dict_f_l  # type: ignore

        return Discretizer(quant_dict)
