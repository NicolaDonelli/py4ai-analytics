"""Model save function."""
import os
import platform
import re
import time
from typing import Any, Dict, Optional

import dill
from py4ai.core.logging import getDefaultLogger
from py4ai.core.utils.fs import mkdir

from py4ai.analytics.ml import PathLike
from py4ai.analytics.ml.core import Transformer


logger = getDefaultLogger()


# TODO [ND] check if this function should be actually kept since there is more a model report class.
def saveModelResults(
    model: Transformer,
    report,  # TODO [ND] type this input
    nameModel: Optional[str] = None,
    suffix: Optional[str] = None,
    modelsPath: str = os.path.join("..", "data", "models"),
    overwrite: bool = False,
):
    """
    Save results of the model.

    :param model: model used
    :param report: report of the model
    :param nameModel: model name used in the path. If None it will valued as <call-timestamp><model-class>_<suffix>
    :param suffix: suffix to be used in the model il nameModel is not specified
    :param modelsPath: models path
    :param overwrite: whether existing files should be overwritten

    :return: path
    """
    if not os.path.isdir(modelsPath):
        mkdir(modelsPath)

    if nameModel is None:
        modelClass = re.sub(r".*\.", "", str(model.__class__))[:-2]
        timestamp = time.strftime("%Y%m%d_%H:%M:%S_")
        suffix = ("_" + suffix) if suffix is not None else ""
        nameModel = timestamp + modelClass + suffix

    return _saveAs(
        {"model": model, "report": report},
        os.path.join(modelsPath, str(nameModel)),
        overwrite=overwrite,
    )


def _saveAs(
    d: Dict[str, Any], modelPath: PathLike, overwrite: bool = False
) -> PathLike:
    if os.path.exists(modelPath):
        if not overwrite:
            raise Exception(f"Model directory [{modelPath}] already exists. ")
    else:
        if platform.system().lower() == "windows" and isinstance(modelPath, str):
            modelPath = modelPath.replace(":", ".")
        os.mkdir(modelPath)

    log = logger(__name__)

    log.info(f" Saving outputs to {modelPath}")
    for k, v in d.items():
        log.debug(f" ---> saving {k}")
        with open(os.path.join(modelPath, k), "wb") as fid:
            dill.dump(v, fid)
    return modelPath
