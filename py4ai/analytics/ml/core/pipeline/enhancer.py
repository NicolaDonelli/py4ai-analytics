"""Implementation of PipelineEnhancer class."""
from typing import List, Union, cast, TypeVar

from py4ai.analytics.ml.core import Enhancer
from py4ai.analytics.ml.core.pipeline import Pipeline, Step
from py4ai.analytics.ml.eda import compose
from py4ai.data.model.text import Document, DocumentsUtilsMixin

TDocumentsUtilsMixin = TypeVar("TDocumentsUtilsMixin", bound=DocumentsUtilsMixin)


class PipelineEnhancer(Enhancer, Pipeline):
    """Enhancer constituted by a sequence of Enhancers."""

    def __init__(self, steps: List[Step]) -> None:
        """
        Class initializer.

        :param steps: list of Enhancer instances that compose the pipeline.
        """
        super(PipelineEnhancer, self).__init__(steps=steps)

    def _validate_steps(self) -> None:
        """
        Check if all steps are Enhancers.

        :raises TypeError: if any of the steps is not an Enhancer
        """
        if any([not isinstance(stage[1], Enhancer) for stage in self.steps]):
            raise TypeError("Each step of the pipeline must be an Enhancer")

    def apply(
        self, documents: Union[Document, TDocumentsUtilsMixin]
    ) -> Union[Document, TDocumentsUtilsMixin]:
        """
        Sequentially enhance documents.

        :param documents: documents to enhance
        :return: enhanced documents
        """
        composer = compose(
            *list(
                map(
                    lambda x: lambda y: cast(Enhancer, x).enhance(y),
                    reversed(self.steps),
                )
            )
        )
        return composer(object)
