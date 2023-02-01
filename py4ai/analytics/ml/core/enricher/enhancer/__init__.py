"""Implementation of abstract enhancer classes."""

from abc import ABC, abstractmethod
from typing import Any

from py4ai.data.model.text import CachedDocuments, Document

from py4ai.analytics.ml.core import Enhancer


class DocumentEnhancher(Enhancer, ABC):
    """Document enhancer abstract class."""

    _type = Document

    @property
    @abstractmethod
    def key(self) -> Any:
        """Key."""
        ...


class CachedDocumentsEnhancher(Enhancer, ABC):
    """CachedDocuments enhancer abstract class."""

    _type = CachedDocuments

    @property
    @abstractmethod
    def key(self) -> Any:
        """Key."""
        ...
