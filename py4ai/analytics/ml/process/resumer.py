"""
Class to get the top n models from the runner results.

The class seems not used by any other module
"""
from py4ai.analytics.ml.process.runner import RunnerResults


# TODO [ND] this class does not seem to do anything. Should we really keep it as it is?
class TopModelsFromRunner(object):
    """Class to get the top n models from the runner results."""

    def __init__(self, model_results: RunnerResults, top: int) -> None:
        """
        Initialize the class.

        :param model_results: instance of RunnerResults.
        :param top: top n runs.
        """
        self.model_results = model_results
        self.top = top
