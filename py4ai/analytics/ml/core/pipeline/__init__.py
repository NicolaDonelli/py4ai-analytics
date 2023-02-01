"""Pipeline module."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Tuple, Union

import six

from py4ai.analytics.ml import ParamMixing
from py4ai.analytics.ml.core import Enhancer, Estimator, Resampler, Transformer

Step = Tuple[str, Union[Estimator, Transformer, Resampler, Enhancer]]


class BaseComposition(Estimator, ABC):
    """Handle parameter management for classifiers composed of named estimators."""

    def _get_params(self, attr: str, deep: bool = True) -> Dict[str, Any]:
        out = super(BaseComposition, self).get_params(deep=deep)
        if not deep:
            return out
        estimators = getattr(self, attr)
        out.update(estimators)
        for name, estimator in estimators:
            if hasattr(estimator, "get_params"):
                for key, value in six.iteritems(estimator.get_params(deep=True)):
                    out["%s__%s" % (name, key)] = value
        return out

    def _set_params(self, attr: str, **params: Any) -> "BaseComposition":
        # Ensure strict ordering of parameter setting:
        # 1. All steps
        if attr in params:
            setattr(self, attr, params.pop(attr))
        # 2. Step replacement
        items = getattr(self, attr)
        names = []
        if items:
            names, _ = zip(*items)
        for name in list(six.iterkeys(params)):
            if "__" not in name and name in names:
                self._replace_estimator(attr, name, params.pop(name))
        # 3. Step parameters and other initialisation arguments
        super(BaseComposition, self).set_params(**params)
        return self

    def _replace_estimator(self, attr: str, name: str, new_val: str) -> None:
        # assumes `name` is a valid estimator name
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)

    def _validate_names(self, names: List[str]) -> None:
        """
        Check consistency of the input list of names.

        Names must be unique, must not conflict with constructor arguments and must not contain __
        :param names: list of names

        :raises ValueError: if names are not unique, or they conflict with constructor arguments, or contain __
        """
        if len(set(names)) != len(names):
            raise ValueError("Names provided are not unique: {0!r}".format(list(names)))
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError(
                "Estimator names conflict with constructor arguments: {0!r}".format(
                    sorted(invalid_names)
                )
            )
        invalid_names = {name for name in names if "__" in name}
        if invalid_names:
            raise ValueError(
                "Estimator names must not contain __: got {0!r}".format(invalid_names)
            )


class Pipeline(ParamMixing, ABC):
    """Abstract class to define pipelines."""

    def __init__(self, steps: List[Step]) -> None:
        """
        Class instance initializer.

        :param steps: list of couples (name, object) constituting the steps of the pipeline
        """
        self._steps = list(self.flatten(steps))
        self._validate_names()
        self._validate_steps()

    @property
    def steps(self) -> List[Step]:
        """
        Pipeline steps.

        :return: pipeline steps
        """
        return self._steps

    @abstractmethod
    def _validate_steps(self) -> None:
        """Not implemented."""
        ...

    def _validate_names(self) -> None:
        """
        Check consistency of the list of names given to self.steps.

        Names must be unique, must not conflict with constructor arguments and must not contain __

        :raises ValueError: if names are not unique or contain __
        """
        names, _ = zip(*self.steps)
        if len(set(names)) != len(names):
            raise ValueError("Names provided are not unique: {0!r}".format(list(names)))
        invalid_names = [name for name in names if "__" in name]
        if invalid_names:
            raise ValueError(
                "Estimator names must not contain __: got {0!r}".format(invalid_names)
            )

    @classmethod
    def flatten(cls, steps: List[Step]) -> Iterable[Step]:
        """
        Flatten pipelines in steps.

        :param steps: list of couples (name, object) constituting the steps of the pipeline

        :yield: flattened steps list
        """
        for x in steps:
            if not isinstance(x[1], Pipeline):
                yield x
            else:
                for (name, step) in cls.flatten(x[1].steps):
                    yield f"{x[0]}_{name}", step
