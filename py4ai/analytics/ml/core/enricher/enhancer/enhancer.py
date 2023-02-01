"""Implementation of classes extending Enhancer."""

from py4ai.data.model.text import CachedDocuments, Document, DocumentsUtilsMixin
from langdetect import detect
from typing import TypeVar

from py4ai.analytics.ml.core.enricher.enhancer import (
    CachedDocumentsEnhancher,
    DocumentEnhancher,
)


TDocumentsUtilsMixin = TypeVar("TDocumentsUtilsMixin", bound=DocumentsUtilsMixin)


class Doc2DocsWrappers(CachedDocumentsEnhancher):
    """Enhance documents in collection with the same document enhancer."""

    def __init__(self, enancher: DocumentEnhancher) -> None:
        """
        Class instance initializer.

        :param enancher: instance to enhance documents
        """
        self.enancher = enancher

    def apply(self, documents: TDocumentsUtilsMixin) -> CachedDocuments:
        """
        Enhance documents in collection with input enhancer.

        :param documents: documents collection
        :return: enhanced documents
        """
        return CachedDocuments(
            list(self.enancher.enhance(doc) for doc in documents.documents)
        )


class LanguageEnhancher(DocumentEnhancher):
    """Add language property to each document."""

    @property
    def key(self) -> str:
        """
        Get field name.

        :return: field name
        """
        return self.field_name

    def __init__(self, field_name: str = "language") -> None:
        """
        Enhance documents in collection with input enhancer.

        :param field_name: name of the field to add to the document
        """
        self.field_name = field_name

    def apply(self, document: Document) -> Document:
        """
        Add field with language specification to document.

        :param document: input document
        :return: enhanced document
        """
        return document.addProperty(self.field_name, detect(document.text))
