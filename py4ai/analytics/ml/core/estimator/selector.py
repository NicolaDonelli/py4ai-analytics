"""Implementation of estimators that select features."""
from typing import List, Optional, Tuple

import dask.dataframe as daskdf
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as stats
from py4ai.data.model.ml import PandasDataset
from dask.multiprocessing import get
from psutil import cpu_count
from typeguard import typechecked
from typing_extensions import Literal

from py4ai.analytics.ml.core import Estimator, Numeric
from py4ai.analytics.ml.core.enricher.transformer.discretizer import toMultiLevels
from py4ai.analytics.ml.core.enricher.transformer.selector import FeatureSelector
from py4ai.analytics.ml.eda.general import computeCorrelation, filterColumnsWith

Correlations = Literal["absolute", "positive", "negative"]


class DeleteConstantFeatures(Estimator):
    """Remove features with ratio between standard deviation and mean less than a defined threshold."""

    def __init__(self, threshold: Numeric = 0) -> None:
        """
        Class instance initializer.

        :param threshold: threshold for the standard deviation of features
        """
        self.threshold = threshold

    @typechecked
    def train(self, dataset: PandasDataset) -> FeatureSelector:
        """
        Select columns to keep and drop.

        :param dataset: Dataset instance with features and labels

        :return: FeatureSelector Transformer
        """
        X = dataset.features
        self.logger.debug("Deleting constant features...")
        to_drop = X.columns[X.std() <= self.threshold]
        to_keep = list(set(X.columns).difference(set(to_drop)))

        self.logger.info(
            "Number of columns dropped by EnhancedDeleteCostantFeatures: %d"
            % len(to_drop)
        )
        self.logger.debug(
            "EnhancedDeleteCostantFeatures dropped columns:\n%s"
            % ("\n".join(sorted(to_drop)))
        )

        return FeatureSelector(to_keep=to_keep, estimator=self)


class FilterOutNans(Estimator):
    """Drop columns with a number of nulls bigger than the selected threshold."""

    def __init__(self, percentage: Numeric = 0.05):
        """
        Class instance initializer.

        :param percentage: minimum percentage of NAs allowed in a column
        """
        self.percentage = percentage

    @typechecked
    def train(self, dataset: PandasDataset) -> FeatureSelector:
        """
        Select columns to keep and drop.

        :param dataset: Dataset instance with features and labels
        :return: FeatureSelector Transformer
        """
        X = dataset.features
        nans = X.isnull().sum() * 1.0 / X.shape[0]
        to_drop = list(nans[nans > self.percentage].index)
        to_keep = list(set(X.columns).difference(set(to_drop)))

        return FeatureSelector(to_keep=to_keep, estimator=self)


class SelectByCorrelation(Estimator):
    """Select features based on a correlation threshold."""

    _correlation_functions = {
        "absolute": lambda x: np.abs(x),
        "positive": lambda x: x,
        "negative": lambda x: -x,
    }

    def __init__(
        self,
        threshold: Numeric = 0.4,
        correlation: Correlations = "absolute",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ):
        """
        Class instance initializer.

        :param threshold: minimum correlation threshold to select variables
        :param correlation: function to apply to calculate correlation one of "absolute", "positive" or "negative". Default "absolute"
        :param start: start date to apply correlation function
        :param end: end date to apply correlation function
        """
        self.threshold = threshold
        self.start = start
        self.end = end
        self._f = self._correlation_functions[correlation]

    @typechecked
    def train(self, dataset: PandasDataset) -> FeatureSelector:
        """
        Select columns to keep and drop.

        :param dataset: Dataset instance with features and labels
        :return: FeatureSelector Transformer
        """
        x = dataset.features
        y = dataset.labels

        if type(x) == pd.DataFrame:
            x_ = x[slice(self.start, self.end)].copy()
        else:
            x_ = pd.DataFrame(x)[slice(self.start, self.end)]

        if type(y) == pd.DataFrame:
            y_ = y[slice(self.start, self.end)].copy()
        else:
            y_ = pd.DataFrame(y, index=x.index)[slice(self.start, self.end)]

        correlations = computeCorrelation(x_, y_).applymap(self._f)

        # Max correlations along rows
        maxCorrelations = correlations.max(axis=1)
        self.correlations = maxCorrelations

        to_keep = maxCorrelations[maxCorrelations > self.threshold].index

        # self.logger.info("Number of columns dropped by SelectByCorrelation: %d" % len(to_drop))
        # self.logger.debug("SelectByCorrelation dropped columns:\n%s" % ("\n".join(sorted(to_drop))))

        return FeatureSelector(to_keep=to_keep, estimator=self)


class SelectorFromSubstrings(Estimator):
    """Select columns with given substings in their name."""

    def __init__(self, strlist: List[str]):
        """
        Class instance initializer.

        :param strlist: list of strings
        """
        self.strlist = strlist

    @typechecked
    def train(self, dataset: PandasDataset) -> FeatureSelector:
        """
        Select columns to keep and drop.

        :param dataset: Dataset instance with features and labels
        :return: FeatureSelector Transformer
        """
        X = dataset.features
        to_keep = filterColumnsWith(X, self.strlist).columns

        # self.logger.info("Number of columns dropped by SelectFromSubstrings: %d" % len(to_drop))
        # self.logger.debug("SelectorFromSubstrings dropped columns:\n%s" % ("\n".join(sorted(to_drop))))

        return FeatureSelector(to_keep=to_keep, estimator=self)


class SelectByKL(Estimator):
    """Select features depending on the Kullback-Leibler divergence.

    Select features depending on the Kullback-Leibler divergence between the distribution of under the two levels of the
    target variable. The ratio is to drop features that to not exhibit particularly different distribution under the two
    levels of the target variables.
    """

    def __init__(
        self, kl_thresh: Numeric = 1.0, y_thresh: Optional[Numeric] = None
    ) -> None:
        """
        Class instance initializer.

        :param kl_thresh: threshold on KL divergence. Drop columns if KL divergence is greater than thresh
        :param y_thresh: threshold to discretize the target variable (in case it is not already discrete)
        """
        self.kl_thresh = kl_thresh
        self.y_thresh = y_thresh
        self.divergences01: Optional[PandasDataset] = None

    @staticmethod
    def _compute_divergences(x: pd.Series, y: pd.Series) -> float:
        """
        Compute KL between x series when y is 0 and when y is 1.

        :param x: feature series
        :param y: target series
        :return: KL divergence
        """
        yvals = np.unique(y)
        points = np.linspace(x.min(), x.max(), max((x.max() - x.min()) * 2, 100))

        if x.std() == 0:  # If the feature is constant, we drop it
            out = 0.0
        elif len(np.unique(x[y == yvals[0]])) == 1:
            # If the feature is constant where the target is not interesting it means that it is not constant where the
            # target is interesting and thus we keep it
            out = np.inf
        elif len(np.unique(x[y == yvals[1]])) == 1:
            # If the feature is constant where the target is interesting it means that it is not constant where the
            # target is not interesting.
            # Thus we check if that value is common where the target is not interesting and, if it is, we drop it.
            if (
                sum(x[y == yvals[0]] == np.unique(x[y == yvals[1]])[0])
                * 1.0
                / len(x[y == yvals[0]])
                > 0.3
            ):
                out = 0.0
            else:
                out = np.inf
        else:
            out = stats.entropy(
                stats.kde.gaussian_kde(x[y == yvals[0]].values.ravel())(points),
                stats.kde.gaussian_kde(x[y == yvals[1]].values.ravel())(points),
            )

        return out

    @typechecked
    def train(self, dataset: PandasDataset) -> FeatureSelector:
        """
        Select columns to keep and drop.

        :param dataset: Dataset instance with features and labels
        :return: FeatureSelector Transformer
        """
        cleaned = dataset.dropna().intersection()

        if len(cleaned) != len(dataset):
            self.logger.warning(
                f"{type(self)}: Training Set has nan values which has been dropped from the training"
            )

        X, y = cleaned.features, cleaned.labels

        if self.y_thresh is not None:
            y = toMultiLevels(y, self.y_thresh)

        if len(np.unique(y)) == 2:
            dX = daskdf.from_pandas(X.T, npartitions=cpu_count() * 2)
            self.divergences01 = dX.apply(
                (lambda x: self._compute_divergences(x, y)), axis=1
            ).compute(get=get)
            to_drop = list(
                self.divergences01[self.divergences01.values < self.kl_thresh].index
            )
        else:
            to_drop = []
        to_keep = list(set(X.columns).difference(to_drop))

        return FeatureSelector(to_keep=to_keep, estimator=self)


class DeleteDuplicates(Estimator):
    """Keep only the most correlated with the target of a set of highly correlated features."""

    def __init__(
        self,
        threshold: Numeric = 0.9,
        min_threshold: Numeric = 0.8,
        step: Numeric = 0.1,
    ):
        """
        Class instance initializer.

        :param threshold: minimum value to consider features to be deleted
        :param min_threshold: min_threshold for graph correlation
        :param step: step used to increase threshold
        """
        self.threshold = threshold
        self.min_threshold = min_threshold
        self.step = step

    def _most_correlated(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[str, float]:
        """
        Calculate the feature with the max correlation.

        :param X: pd.DataFrame with features
        :param y: pd.DataFrame with targets
        :return: tuple with the name of the variable and the value of correlation
        """
        corr = computeCorrelation(X, y).abs().max(axis=1)
        ret = (corr.idxmax(), corr.max())
        if len(corr) > 1:
            self.logger.info(
                f"Keeping \033[1m{ret[0]}\033[0m [{round(ret[1], 2)}] between {', '.join(corr.index)}"
            )
        return ret

    @staticmethod
    def _least_correlated_components(
        corr: pd.DataFrame, cols: List[str]
    ) -> Tuple[float, int]:
        """
        Calculate least correlated features.

        :param corr: corr matrix
        :param cols: columns to include in calculation
        :return: tuple with the name of the variable, the value of correlation and the number of cols considered
        """
        sub_corr = corr[cols].loc[cols].abs().min(axis=1)
        return sub_corr.min(), len(cols)

    def _graph_correlation(self, dataset: PandasDataset) -> PandasDataset:
        """
        Get list with columns to drop based on the thresholds selected.

        :param dataset: input dataset
        :return: new dataset with removed columns
        """
        corrs = computeCorrelation(dataset.features).abs()
        thres = self.threshold
        self.logger.info(f"Initial pruning threshold: {self.threshold}")
        to_keep = dataset.features.columns

        check = False
        ith = 0
        while not check:
            ith += 1
            self.logger.debug("Iteration: %d" % ith)
            # Undirected Graph with edges given by correlations over threshold
            mat = corrs.applymap(lambda x: 1 if x > thres else 0)
            g = nx.Graph(mat.values)
            for icol, col in enumerate(mat.columns):
                g.nodes[icol].update({"name": col})

            # List of couples of least correlated features for each connected component
            minCorrs = [
                self._least_correlated_components(
                    corrs, [g.nodes[x]["name"] for x in group]
                )
                for group in nx.connected_components(g)
            ]

            self.logger.info(
                f"Number of non-singleton connected components: "
                f"{sum(map(lambda x: 1 if x[1] > 1 else 0, minCorrs))}"
            )
            self.logger.info(
                f"Total elements in non-sigleton connected components: "
                f"{sum(map(lambda x: x[1] if x[1] > 1 else 0, minCorrs))}"
            )
            self.logger.info(
                f"Number of connected components with minimal correlation under {self.min_threshold}: "
                f"{len([t for t in minCorrs if t[0] <= self.min_threshold])}"
            )

            # Check that all correlations are above a certain threshold
            check = all(map(lambda x: x[0] > self.min_threshold, minCorrs))
            self.logger.debug("check: " + str(check))

            if not check:
                thres = (1 - self.step) * thres + self.step
                self.logger.info(f"Updating threshold to: {thres}\n")
            else:
                to_keep = [
                    self._most_correlated(
                        dataset.features[[g.nodes[x]["name"] for x in group]],
                        dataset.labels,
                    )[0]
                    for group in nx.connected_components(g)
                ]
                self.logger.debug(f"Number of columns to keep: {len(to_keep)}")

        if set(to_keep) != set(dataset.features.columns):
            return self._graph_correlation(
                dataset.createObject(dataset.features[to_keep], dataset.labels)
            )
        else:
            return dataset

    @typechecked
    def train(self, dataset: PandasDataset) -> FeatureSelector:
        """
        Get the list of columns to drop based on the correlation function selected.

        :param dataset: input data
        :return: feature selector instance
        """
        outdataset = self._graph_correlation(
            DeleteConstantFeatures(threshold=0).train(dataset).transform(dataset)
        )

        self.logger.info(
            f"Number of columns to keep: {len(outdataset.features.columns)}"
        )
        self.logger.debug(
            "Columns to keep:\n%s" % ("\n".join(sorted(outdataset.features.columns)))
        )

        return FeatureSelector(
            to_keep=list(outdataset.features.columns), estimator=self
        )
