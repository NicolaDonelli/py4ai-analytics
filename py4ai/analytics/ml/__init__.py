"""Analytics core module."""

from __future__ import absolute_import

import os
from abc import ABC
from collections import defaultdict
from inspect import signature
from typing import Type, List, Dict, Any, Union, Sequence, TypeVar

import pandas as pd
from py4ai.core.logging import WithLogging
from numpy import ndarray
from sklearn import clone
from typing_extensions import Literal

InputType = Literal["pandas", "dict", "list", "array"]
ArrayLike = Union[pd.Series, pd.DataFrame, ndarray, Sequence]
PathLike = Union[str, "os.PathLike[str]"]


class InputTypeSetter(ABC):
    """
    Set the type of feature space and label to be extracted form datasets.

    Available types are "pandas", "dict", "list" and "array"
    """

    _input_type: Literal["pandas", "dict", "list", "array"] = "pandas"

    @property
    def input_type(self) -> InputType:
        """
        Input type.

        :return: input type
        """
        return self._input_type

    @input_type.setter
    def input_type(self, value: InputType) -> None:
        """
        Type of feature space and label to be extracted form datasets.

        :param value: type of feature space and label to be extracted form datasets.
            Available types are "pandas", "dict", "list" and "array"
        """
        self.set_input_type(value)

    def set_input_type(self, value: InputType) -> "InputTypeSetter":
        """
        Set the type of feature space and label to be extracted form datasets.

        :param value: type of feature space and label to be extracted form datasets.
            Available types are "pandas", "dict", "list" and "array"
        :return: self
        """
        self._input_type = value
        return self


TParamMixing = TypeVar("TParamMixing", bound="ParamMixing", covariant=True)


class ParamMixing(WithLogging, ABC):
    """Param mixing class."""

    @classmethod
    def _get_param_names(cls: Type[TParamMixing]) -> List[str]:
        """
        Get parameter names for the estimator.

        :raises RuntimeError: if the estimator does not specify the parameters in the signature
        :return: list of parameters
        """
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    f"scikit-learn estimators should always specify their parameters in the signature"
                    f"of their __init__ (no varargs). {cls} with constructor {init_signature} doesn't "
                    f"follow this convention."
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        :param deep: boolean, optional. If True, will return the parameters for this estimator and contained sub-objects
            that are estimators.

        :return: mapping of string to any parameter names mapped to their values.
        """
        out: Dict[str, Any] = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()  # type: ignore
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self: TParamMixing, **params: Any) -> TParamMixing:
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects (such as pipelines).
        The latter have parameters of the form ``<component>__<parameter>`` so that it's possible
        to update each component of a nested object.

        :param params: keyworded parameters

        :raises ValueError: if one of the provided parameters does not exist

        :return: self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params: dict = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def clone(self: TParamMixing) -> TParamMixing:
        """
        Return a clone of self.

        :return: clone of self
        """
        return clone(self)
