"""Implementation of the SubselectionEstimator."""
from typing import List, Optional

from py4ai.data.model.ml import PandasDataset

from py4ai.analytics.ml.core import Estimator
from py4ai.analytics.ml.core.enricher.transformer.selector import (
    FeatureSelector,
    LabelSelector,
)
from py4ai.analytics.ml.core.pipeline.transformer import (
    PipelineEstimator,
    PipelineTransformer,
)


class SubselectionEstimator(Estimator):
    """
    Create an estimator that operates on a subselection of a multi-index PandasDataset.

    Subselection can be done either on the features or on the labels, or both
    """

    def __init__(
        self,
        estimator: Estimator,
        features_name: Optional[List[str]] = None,
        labels_name: Optional[List[str]] = None,
    ):
        """
        Class instance initializer.

        :param estimator: Generic estimators that operates on PandasDataset
        :param features_name: Name of the level of the multi-index PandasDataset for the features to be selected
        :param labels_name: Name of the level of the multi-index PandasDataset for the labels to be selected
        """
        self.features_name = features_name
        self.labels_name = labels_name
        self.estimator = estimator

    @property
    def pipeline(self) -> PipelineEstimator:
        """
        Generate the pipeline estimator corresponding to this estimator.

        :return: PipelineEstimator
        """
        return PipelineEstimator(
            steps=(
                [("labelSelector", LabelSelector(self.labels_name))]
                if self.labels_name is not None
                else []
            )
            + (
                [("featureSelector", FeatureSelector(self.features_name))]
                if self.features_name is not None
                else []
            )
            + ([("est", self.estimator)])
        )

    def train(self, dataset: PandasDataset) -> PipelineTransformer:
        """
        Train the estimator on the given dataset.

        :param dataset: input dataset. It contains both train features and train labels, already aligned.

        :return: Trained model
        :raises TypeError: dataset is not a PandasDataset
        """
        if not isinstance(dataset, PandasDataset):
            raise TypeError("Input dataset must be of type PandasDataset")

        self.estimator.logger.setLevel("CRITICAL")

        return self.pipeline.train(dataset)
