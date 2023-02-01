"""Implementation of classes related to coherence measures."""
import statistics as stat
from abc import ABC, abstractmethod
from collections import namedtuple
from itertools import chain, combinations, product
from math import log as log_e
from typing import Callable, Dict, Iterable, Optional, Tuple, Union, List, Sequence, Set, Any, cast

import networkx as nx
import numpy as np
import pandas as pd
from cachetools import LFUCache, cachedmethod
from py4ai.data.model.ml import TDatasetUtilsMixin, PandasDataset
from py4ai.core.logging import WithLogging
from py4ai.core.utils.decorators import lazyproperty
from py4ai.core.utils.dict import groupBy, union
from numpy import argwhere, dot
from numpy.linalg import norm
from numpy.typing import NDArray
from seaborn import catplot, relplot
from typeguard import typechecked

from py4ai.analytics.ml.core import Evaluator, Metric
from py4ai.analytics.ml.core import Numeric
from py4ai.analytics.ml.core.pipeline.transformer import PipelineTransformer


def jaccardIndex(x: set, y: set) -> Numeric:
    """
    Jaccard index between two sets.

    :param x: first vector
    :param y: second vector
    :return: Jaccard index
    """
    return len(x.intersection(y)) / len(x.union(y)) if len(x.union(y)) > 0 else 0


def v_cosine(x: NDArray[Numeric], y: NDArray[Numeric]) -> Numeric:
    """
    Cosine of the angle between two arrays.

    :param x: first vector
    :param y: second vector
    :return: v cosine
    """
    norm_prod = norm(x) * norm(y)
    return dot(x, y) / norm_prod if norm_prod != 0 else 0


def my_sd(x: NDArray[Numeric]) -> Numeric:
    """
    Return 0 if the length of the array <= 1, otherwise the standard deviation of the values.

    :param x: vector
    :return: standard deviation if len(x) > 1 else 0
    """
    return 0 if len(x) <= 1 else stat.stdev(x)


def findsubsets(iterable: Sequence, r: int) -> List[List[str]]:
    """
    Return the list of all subsets of given length.

    :param iterable: data
    :param r: number of elements in the subsets
    :return: list of lists representing subsets of the original sequence
    """
    return list(map(list, combinations(iterable, r)))


def findAllSubset(iterable: Sequence, entire: bool = True) -> List[List[str]]:
    """
    Find all subsets of any length.

    :param iterable: data
    :param entire: includes also the entire set
    :return: list of all subsets of any length
    """
    return list(
        chain(
            *map(
                lambda n: findsubsets(iterable, n),
                range(1, len(iterable) + 1 if entire else len(iterable)),
            )
        )
    )


# TODO: Segmentation could became List[Tuple[Set, Set]]
Segmentation = List[Tuple[List, List]]
segmentationsFuns = {
    "S_one_one": lambda x: list(combinations(findsubsets(x, 1), 2))
    + list(combinations(findsubsets(x[::-1], 1), 2)),
    "S_one_pre": lambda x: list(combinations(findsubsets(x, 1), 2)),
    "S_one_post": lambda x: list(combinations(findsubsets(x[::-1], 1), 2)),
    "S_one_any": lambda x: [
        (W, W_star)
        for W, W_star in product(findsubsets(x, 1), findAllSubset(x, entire=False))
        if W[0] not in W_star
    ],
    "S_one_all": lambda x: [
        (W, W_star)
        for W, W_star in product(findsubsets(x, 1), findsubsets(x, len(x) - 1))
        if W[0] not in W_star
    ],
    "S_one_set": lambda x: [(W, x) for W in findsubsets(x, 1)],
}

MeasureSpec = namedtuple("MeasureSpec", ["type", "spec"])


class CoherenceMeasure(ABC):
    """Abstract class representing a coherence measure."""

    name: str
    spec: MeasureSpec

    @property
    def key(self):
        """
        Return the key of this object.

        :return: key of this object
        """
        return f"{self.spec.type}-{self.spec.spec}"

    @abstractmethod
    def compute(self, prob: Callable[[List], float], segmentation: Segmentation) -> List[float]:
        """
        Compute the distance.

        :param prob: probability
        :param segmentation: segmentation
        :return: computation
        """
        raise NotImplementedError


class DirectMeasures(CoherenceMeasure, ABC):
    """Direct measures."""

    def compute(self, prob: Callable[[List], float], segmentation: Segmentation) -> List[float]:
        """
        Compute the distance.

        :param prob: probability
        :param segmentation: segmentation
        :return: computation
        """
        return [self.confirmation(prob, sn[0], sn[1]) for sn in segmentation]

    @abstractmethod
    def confirmation(self, prob: Callable[[List], float], s1: List, s2: List) -> float:
        """
        Compute confirmation.

        :param prob: probability
        :param s1: s1
        :param s2: s2
        :return: confirmation
        """
        raise NotImplementedError


class M_d(DirectMeasures):
    """M_d measures."""

    name: str = "m_d"
    spec: MeasureSpec = MeasureSpec("direct", "conditional")

    def confirmation(self, prob: Callable[[List], float], s1: List, s2: List) -> float:
        """
        Compute confirmation.

        :param prob: probability
        :param s1: s1
        :param s2: s2
        :return: confirmation
        """
        p1, p2, p12 = prob(s1), prob(s2), prob(s1 + s2)
        return p12 / p2 - p1 if (p2 != 0) else 0.0


class LogMeasures(DirectMeasures):
    """Log measures."""

    spec: MeasureSpec = MeasureSpec("direct", "log_ratio")

    def __init__(self, epsilon: float = pow(10, -12)):
        """
        Class instance initializer.

        :param epsilon: epsilon
        """
        assert epsilon > 0
        self.epsilon = epsilon


class M_lr(LogMeasures):
    """M_lr measures."""

    name: str = "m_lr"

    def confirmation(self, prob: Callable[[List], float], s1: List, s2: List) -> float:
        """
        Compute confirmation.

        :param prob: probability
        :param s1: s1
        :param s2: s2
        :return: confirmation
        """
        p1, p2, p12 = prob(s1), prob(s2), prob(s1 + s2)
        return log_e((p12 + self.epsilon) / (p2 * p1 + self.epsilon))


class M_nlr(LogMeasures):
    """M_nlr measures."""

    name: str = "m_nlr"

    def confirmation(self, prob: Callable[[List], float], s1: List, s2: List) -> float:
        """
        Compute confirmation.

        :param prob: probability
        :param s1: s1
        :param s2: s2
        :return: confirmation
        """
        p1, p2, p12 = prob(s1), prob(s2), prob(s1 + s2)
        return -(log_e((p12 + self.epsilon) / (p2 * p1 + self.epsilon))) / log_e(
            p12 + self.epsilon
        )


class M_lc(LogMeasures):
    """M_lc measures."""

    name: str = "m_lc"

    def confirmation(self, prob: Callable[[List], float], s1: List, s2: List) -> float:
        """
        Compute confirmation.

        :param prob: probability
        :param s1: s1
        :param s2: s2
        :return: confirmation
        """
        p2, p12 = prob(s2), prob(s1 + s2)
        return log_e((p12 + self.epsilon) / (p2 + self.epsilon))


class IndirectMeasure(CoherenceMeasure):
    """Indirect measure."""

    def __init__(
        self, measure: DirectMeasures, similarity: Callable[[NDArray[float], NDArray[float]], float], gamma: float = 1.0
    ):
        """
        Class instance initializer.

        :param measure: measure
        :param similarity: function that calculates the similarity
        :param gamma: gamma
        """
        self.measure = measure
        self.similarity = similarity
        self.gamma = gamma

    def getVector(self, prob: Callable[[List], float], s1: List, w: List) -> NDArray[float]:
        """
        Get vector.

        :param prob: probability
        :param s1: s1
        :param w: w
        :return: vector
        """
        return np.array(
            [
                sum(
                    [
                        self.measure.confirmation(prob, [wi], [wj]) ** self.gamma
                        for wi in s1
                    ]
                )
                for wj in w
            ]
        )

    def compute(self, prob: Callable, segmentation: Iterable) -> List[float]:
        """
        Compute the distance.

        :param prob: probability
        :param segmentation: segmentation
        :return: computation
        """
        ws = [
            fs
            for fs in set.union(
                *([set(w) for ws in segmentation for w in ws] + [set()])
            )
        ]

        return [
            self.similarity(
                self.getVector(prob, sn[0], ws), self.getVector(prob, sn[1], ws)
            )
            for sn in segmentation
        ]


class M_cos_nlr(IndirectMeasure):
    """M_cos_nlr measure."""

    name: str = "m_cos_nlr"
    spec: MeasureSpec = MeasureSpec("indirect", "log_ratio")

    def __init__(self, gamma: float = 1.0, epsilon: float = pow(10, -12)):
        """
        Class instance initializer.

        :param gamma: gamma
        :param epsilon: epsilon
        """
        super(M_cos_nlr, self).__init__(M_nlr(epsilon), v_cosine, gamma)


class SegmentationMeasure(object):
    """Class of the couple segmentation and measure."""

    @typechecked
    def __init__(self, segmentation: str, measure: CoherenceMeasure):
        """
        Class instance initializer.

        :param segmentation: segmentation
        :param measure: measure
        :raises ValueError: if segmentation is not a known one
        """
        self.segmentation = segmentation
        self.measure = measure
        try:
            self.segmentationFun = segmentationsFuns[self.segmentation]
        except KeyError:
            raise ValueError(
                f'"{self.segmentation}" is not an admissible segmentation. '
                f"Admissible segmentations are:"
                f"{set(segmentationsFuns.keys())}"
            )

    def __eq__(self, other: object) -> bool:
        """
        Two instances are equals if both measure and segmentation are equal.

        :param other: other
        :return: true if equal
        """
        if not isinstance(other, SegmentationMeasure):
            return NotImplemented

        return (
            other
            and self.segmentation == other.segmentation
            and self.measure == other.measure
        )

    def __ne__(self, other: object) -> bool:
        """
        Two instances are not equal if either measure and segmentation are not equal.

        :param other: other
        :return: true if not equal
        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """
        Calcuate hash of this object.

        :return: hash
        """
        return hash((self.segmentation, self.measure))


class SegmentationMeasureSet(frozenset):
    """Segmentation measure set."""

    def __init__(self, *args, **kwds):
        """
        Class instance initializer.

        :param args: not used
        :param kwds: not used
        """
        super(SegmentationMeasureSet, self).__init__()
        self._validate()

    def _validate(self):
        for el in self:
            if not isinstance(el, SegmentationMeasure):
                raise ValueError("each element of the set must be a Measure namedTuple")


class Probability(object):
    """Probability."""

    def __init__(self, name: str = "P_bd", sw: int = 10):
        """
        Class instance initializer.

        :param name: name
        :param sw: sw
        """
        self.name = name
        self.sw = sw


class ReductionMethod(object):
    """Reduction method."""

    def __init__(
        self,
        atLeastOneOccurrence: bool = True,
        jaccardSimilarityThreshold: Numeric = 1,
        topn: int = None
    ):
        """
        Class instance initializer.

        :param atLeastOneOccurrence: at least one occurrence
        :param jaccardSimilarityThreshold: threshold for Jaccard similarity
        :param topn: topn
        """
        self.atLeastOneOccurrence = atLeastOneOccurrence
        self.jaccardSimilarityThreshold = jaccardSimilarityThreshold
        self.topn = topn
        self._validate()

    def _validate(self):
        if not isinstance(self.atLeastOneOccurrence, bool):
            raise ValueError(
                'The first argument "atLeastOneOccurrence" must be boolean (True/False)'
            )
        if (type(self.jaccardSimilarityThreshold) not in [int, float]) & (
            self.jaccardSimilarityThreshold is not None
        ):
            raise ValueError(
                'The second argument "jaccardSimilarityThreshold" must be Numeric or None'
            )
        if ((self.topn is not None) & (not isinstance(self.topn, int))) | isinstance(
            self.topn, bool
        ):
            raise ValueError(
                'The third argument "topn" must be integer or None (boolean value returns error)'
            )


# create default elements
defaultSegmentationMeasureSet = SegmentationMeasureSet(
    {
        SegmentationMeasure("S_one_set", M_cos_nlr()),
        SegmentationMeasure("S_one_any", M_d()),
        SegmentationMeasure("S_one_one", M_cos_nlr()),
        SegmentationMeasure("S_one_one", M_nlr()),
        SegmentationMeasure("S_one_one", M_lr()),
        SegmentationMeasure("S_one_pre", M_lc()),
    }
)

defaultSummaryFunctions = {
    "mean": cast(Callable[[Iterable[Numeric]], Numeric], stat.mean),
    "median": cast(Callable[[Iterable[Numeric]], Numeric], stat.median),
    "sd": cast(Callable[[Iterable[Numeric]], Numeric], my_sd),
    "N marginal": cast(Callable[[Iterable[Numeric]], Numeric], len),
}


class CoherenceEvaluator(Evaluator):
    """Evaluator for choerence metric."""

    @typechecked
    def __init__(
        self,
        corpus: TDatasetUtilsMixin[List[str], Any],
        probability: Probability = Probability(),
        textTransformer: PipelineTransformer = None,
        reductionParams: ReductionMethod = ReductionMethod(True, 1, 10),
        segmentationMeasures: SegmentationMeasureSet = defaultSegmentationMeasureSet,
    ):
        """
        Class instance initializer.

        :param corpus: dataset containing text documents used to evaluate the coherence metrics. The document texts are stored in the features of the datatset
        :param probability: Probability used (boolean document)
        :param textTransformer: text transformer. The Transformer takes a PandasDataset as input and its method transform returns a two columns PandasDataset
        :param reductionParams: parameters for keywords reduction method
        :param segmentationMeasures: measure metric specification
        """
        self.corpus = corpus
        self.textTransformer = textTransformer
        self.probability = probability
        self.reductionParams = reductionParams
        self.segmentationMeasures = segmentationMeasures
        self.corpus = corpus

    @lazyproperty
    def corpusProcessed(self) -> TDatasetUtilsMixin[List[str], Any]:
        """
        Purify document texts using textTransformer and probability.

        :return: transformed dataset
        :raises ValueError: if the provided probability is not implemented
        """
        transformedDataset = (
            self.corpus
            if self.textTransformer is None
            else self.textTransformer.transform(self.corpus)
        )

        if self.probability.name == "P_bd":
            return transformedDataset
        else:
            raise ValueError(
                f'Probability "{self.probability.name}" is not implemented'
            )

    @lazyproperty
    def lenCorpus(self) -> int:
        """
        Return number of documents in the corpus.

        :return: number of documents in the corpus
        """
        return len(self.corpusProcessed)

    @staticmethod
    def isIn(word: str, text: str) -> bool:
        """
        Return mapping of a word in doc True or False.

        :param word: word
        :param text: text

        :return: mapping of a word in doc True or False.
        """
        text_split = text.split(" ")
        return (word in text) & all([k in text_split for k in word.split(" ")])

    def buildKeywordsMapping(self, dataset: TDatasetUtilsMixin[List[str], Any]) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Purify the keywords of the dataset and find the keywords mapping through the documents.

        :param dataset: topic dataset

        :return: dictionary of the purified keywords and the keywords mapping over the documents
        """
        keywords = list(dataset.getFeaturesAs(type="array").flat)
        purified_keywords = (
            list(dataset.getFeaturesAs(type="array").flat)
            if self.textTransformer is None
            else list(
                self.textTransformer.transform(dataset).getFeaturesAs(type="array").flat
            )
        )

        dictPurifiedKey = dict(zip(keywords, purified_keywords))

        mapKeyDoc = {
            key: {
                idx_doc
                for idx_doc in self.corpusProcessed.index
                if self.isIn(
                    dictPurifiedKey[key],
                    self.corpusProcessed.loc(idx_doc)
                    .getFeaturesAs(type="array")
                    .flat[0],
                )
            }
            for key in dictPurifiedKey
        }

        return dictPurifiedKey, mapKeyDoc

    def evaluate(self, dataset: TDatasetUtilsMixin[List[str], Any]) -> 'CoherenceMetric':
        """
        Evaluate the coherence metric.

        :param dataset: Dataset where every sample is one word or a composition of words and the label is a label associated
            to the given clustering to be evaluated. We need to reformat the data we have (topic_id: [keywords]) in a format
            list of "keyword, topic_id", which we want to evaluate the clustering
        :return: Metric Object
        """
        # build keywords mapping
        dictPurifiedKey, mapKeyDoc = self.buildKeywordsMapping(dataset)
        numberOfDoc = len(self.corpusProcessed)

        return CoherenceMetric(
            dataset=dataset,
            numberOfDoc=numberOfDoc,
            dictPurifiedKey=dictPurifiedKey,
            mapKeyDoc=mapKeyDoc,
            reductionParams=self.reductionParams,
            segmentationMeasures=self.segmentationMeasures,
        )


class CoherenceMetric(Metric, WithLogging):
    """CoherenceMetric class."""

    def __init__(
        self,
        dataset: TDatasetUtilsMixin[List[str], Any],
        numberOfDoc: int,
        dictPurifiedKey: Dict[str, str],
        mapKeyDoc: Dict[str, Any],
        reductionParams: ReductionMethod = ReductionMethod(),
        segmentationMeasures: SegmentationMeasureSet = defaultSegmentationMeasureSet,
        mainMetric: Optional[Tuple[str, str]] = None,
        summaryFunctions: Dict[str, Callable[[Iterable[Numeric]], Numeric]] = defaultSummaryFunctions,
        summaryAddUsedKeys: bool = True,
        summaryAddDocNumbers: bool = True,
    ):
        """
        Class instance initializer.

        :param dataset: two column Dataset containing topic information. The labels are the topics ids while the features contains the
        :param numberOfDoc: number of documents used
        :param dictPurifiedKey: dictionary with keywords (dictionary keys) to purified keywords (dictionary values)
        :param mapKeyDoc: dictionary with the mapping of the keywords through documents
        :param reductionParams: parameters used to reduce keywords of each topic. Default value None means that the class inherits the parameters form the evaluetor
        :param segmentationMeasures: measures to be computed. Default value None means that the class inherits the measures from the evaluetor
        :param mainMetric: main metric to be used to evaluate performances. The key of the metric should be referring to the segmentation measure to
            be used: (segmentationSet, measure), e.g. ("S_one_set","m_d"). If None, the overall mean is used.
        :param summaryFunctions: functions to be reported in the summary.
        :param summaryAddUsedKeys: whether or not to add used keys in the summary.
        :param summaryAddDocNumbers: whether or not to add documents number in the summary.
        """
        super(CoherenceMetric, self).__init__()
        self.dataset = dataset
        self.numberOfDoc = numberOfDoc
        self.dictPurifiedKey = dictPurifiedKey
        self.mapKeyDoc = mapKeyDoc
        self.reductionParams = reductionParams
        self.segmentationMeasures = segmentationMeasures
        self.mainMetric = mainMetric
        self.summaryFunctions = summaryFunctions
        self.summaryAddUsedKeys = summaryAddUsedKeys
        self.summaryAddDocNumbers = summaryAddDocNumbers

        self.cache: LFUCache = LFUCache(maxsize=1024)

    def metricWithNewParams(
        self,
        reductionParams: ReductionMethod = None,
        segmentationMeasures: SegmentationMeasureSet = None,
    ) -> Metric:
        """
        Build a new Metric with different hyperparameter (redutionParams and measures).

        :param reductionParams: object of class ReductionMethod. If None, the evaluator parameters are used.
        :param segmentationMeasures: MeasureSet. If None, the evaluator measures are used.

        :return: Metric object
        """
        newRedParams = (
            reductionParams if reductionParams is not None else self.reductionParams
        )
        newSegmMeasures = (
            segmentationMeasures
            if segmentationMeasures is not None
            else self.segmentationMeasures
        )
        return CoherenceMetric(
            self.dataset,
            self.numberOfDoc,
            self.dictPurifiedKey,
            self.mapKeyDoc,
            newRedParams,
            newSegmMeasures,
        )

    @lazyproperty
    def segmentationMeasureDict(self) -> Dict[Any, List[SegmentationMeasure]]:
        """
        Compute segmentation measure dictionary.

        :return: segmentation measure dictionary
        """
        return dict(groupBy(self.segmentationMeasures, key=lambda x: x.segmentation))

    @lazyproperty
    def segmentations(self) -> Set[str]:
        """
        Return set of segmentations used.

        :return: set of segmentations used
        """
        return self.segmentationMeasureDict.keys()

    def makeSegmentation(self, wordSet: Set[str]) -> Dict[str, Segmentation]:
        """
        Create dictionary of segmentations {'segmenatationName': [segmTuples]}.

        :param wordSet: word set

        :return: dictionary of segmentations
        """
        return {s: segmentationsFuns[s](list(wordSet)) for s in self.segmentations}

    @lazyproperty
    def marginalMeasures(self) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        """
        Compute marginal measures.

        :return: marginal measures
        """
        self.logger.info("Compute marginal measures")
        tSegmentations = {
            topic_id: self.makeSegmentation(self.reducedKeys[topic_id])
            for topic_id in self.reducedKeys
        }
        return {
            topic_id: {
                s_name: self._computeMarginalMeasures(
                    segmentation, self.segmentationMeasureDict[s_name]
                )
                for s_name, segmentation in segmentationByTopic.items()
            }
            for topic_id, segmentationByTopic in tSegmentations.items()
        }

    def _getWordSetHash(self, wordSet: Set[str]) -> int:
        return hash(frozenset(wordSet))

    def _computeMarginalMeasures(self, segmentation: Segmentation, sMeasures: List[SegmentationMeasure]) -> Dict[str, List[float]]:
        """
        Compute marginal measures of a segmentation.

        :param segmentation: segmentation
        :param sMeasures: set of measures to be computed
        :return: marginal measures of a segmentation
        """
        return union(
            {
                sm.measure.name: sm.measure.compute(self.probWords, segmentation)
                for sm in sMeasures
            },
            {"segmentation": segmentation},
        )

    # It would be best to consider all wordSet some frozenset, but we can work this out at a later stage

    @cachedmethod(lambda self: self.cache, key=_getWordSetHash)
    def probWords(self, words: List[str]) -> float:
        """
        Calculate words probability.

        :param words: list of reducedKeywords
        :return: words probability
        """
        # words document
        wordsDoc = set.intersection(*[self.unionWordsDocs(tkey) for tkey in words])
        return len(wordsDoc) / self.numberOfDoc

    def unionWordsDocs(self, words: Iterable[str]) -> Set[str]:
        """
        Return the set that map any word in 'words'.

        :param words: iterable of words
        :return: set that map any word in 'words'
        """
        return {id for word in words for id in self.mapKeyDoc.get(word, {})}

    @lazyproperty
    def topics(self) -> Dict[str, Set[NDArray]]:
        """
        Compute topics.

        :return: topics
        """
        all_labels = self.dataset.getLabelsAs(type="pandas").to_numpy()
        all_features = self.dataset.getFeaturesAs(type="pandas").to_numpy()
        return {
            label: {
                f
                for f in all_features[
                    argwhere(all_labels == label)[:, 0],
                    argwhere(all_labels == label)[:, 1],
                ]
            }
            for label in all_labels.flat
        }

    @lazyproperty
    def reducedKeys(self) -> Dict[str, Union[List[Tuple[str, ...]], Set[Tuple[Any]], Set[Tuple[str, ...]]]]:
        """
        Compute reduced keys.

        :return: reduced keys
        """
        # Consider only keywords with at least one match in the evaluetor?
        if self.reductionParams.atLeastOneOccurrence:
            original_topics = {
                topic: {key for key in self.topics[topic] if bool(self.mapKeyDoc[key])}
                for topic in self.topics
            }
        else:
            original_topics = self.topics

        # cluster the keywords using jaccard similarity
        if (self.reductionParams.jaccardSimilarityThreshold is not None) & (
            self.reductionParams.jaccardSimilarityThreshold < 1
        ):
            out = {
                topic: self.jaccardCluster(
                    original_topics[topic],
                    self.reductionParams.jaccardSimilarityThreshold,
                )
                for topic in original_topics
            }
        else:
            out = {
                topic: {(word,) for word in original_topics[topic]}
                for topic in original_topics
            }

        if self.reductionParams.topn is not None:
            out = {
                topic: set(
                    sorted(
                        out[topic],
                        key=lambda x: len(self.unionWordsDocs(x)),
                        reverse=True,
                    )[: self.reductionParams.topn]
                )
                for topic in out
            }

        return out

    def jaccardCluster(self, wordSet: Set[str], threshold: float) \
            -> Union[List[Tuple[str, ...]], Set[Tuple[Any]], Set[Tuple[str, ...]]]:
        """
        Calculate Jaccard cluster.

        :param wordSet: word set
        :param threshold: threshold
        :return: Jaccard cluster
        """
        wordList = list(wordSet)
        wordPurifiedList = [self.dictPurifiedKey[key] for key in wordList]
        mat = np.array(
            [
                [
                    1
                    if jaccardIndex(set(w1.split(" ")), set(w2.split(" "))) > threshold
                    else 0
                    for w2 in wordPurifiedList
                ]
                for w1 in wordPurifiedList
            ]
        )
        return [
            tuple(wordList[g] for g in group)
            for group in nx.connected_components(nx.Graph(mat))
        ]

    @lazyproperty
    def value(self) -> float:
        """
        Return the mean value.

        :return: mean value
        """
        mean = self.summary["mean"].unstack(level=0)
        return (
            mean.loc[self.mainMetric].mean()
            if self.mainMetric is not None
            else mean.mean().mean()
        )

    @property
    def summary(self) -> pd.DataFrame:
        """
        Summary of the datasets.

        :return: summary
        """
        out = pd.DataFrame.from_dict(
            {
                (topic, partition, measure): {
                    name: fun(value) for name, fun in self.summaryFunctions.items()
                }
                for topic, measuresForTopic in self.marginalMeasures.items()
                for partition, measureByPartition in measuresForTopic.items()
                for measure, value in measureByPartition.items()
                if measure != "segmentation" and len(value) > 0
            },
            orient="index",
        )

        if self.summaryAddUsedKeys:
            out["nUsedKeys"] = [len(self.reducedKeys[idx[0]]) for idx in out.index]

        if self.summaryAddDocNumbers:
            out["totalDocNumber"] = [
                len(
                    {
                        id
                        for tkey in self.reducedKeys[idx[0]]
                        for id in self.unionWordsDocs(tkey)
                    }
                )
                for idx in out.index
            ]

        return out

    def catPlot(self, y: str = "mean", kind: str = "box") -> None:
        """
        Return cat Plot of the summary.

        :param y: which aggregation on y axis
        :param kind: type of plot {'box' = boxplot, 'swarm' = scatterplot}
        """
        pdf_tmp = self.summary.rename_axis(
            ["Topic name", "Partition", "Measure"]
        ).reset_index()
        catplot(
            y=y,
            col="Partition",
            row="Measure",
            hue=None,
            kind=kind,
            data=pdf_tmp,
            showmeans=True,
            sharey=False,
            sharex=False,
        )

    def scatterPlot(self, y: str = "mean") -> None:
        """
        Return scatter Plot of the summary.

        :param y: which aggregation
        """
        pdf_tmp = self.summary.rename_axis(
            ["Topic name", "Partition", "Measure"]
        ).reset_index()
        g = relplot(
            x="totalDocNumber",
            y=y,
            col="Partition",
            row="Measure",
            hue="Topic name",
            kind="scatter",
            data=pdf_tmp,
            alpha=0.5,
            size="nUsedKeys",
            sizes=(40, 500),
            facet_kws={"sharey": False, "sharex": False},
        )
        g.set(xscale="log")

    def scatterPlotWithBenchmark(self, y: str = "mean", seed: int = 42) -> None:
        """
        Return scatter Plot of the summary with benchmark.

        :param y: which aggregation
        :param seed: seed
        """
        np.random.seed(seed)
        n_topic = len(self.topics)
        newLabels = pd.Series(
            np.random.choice(
                [f"benchMark_{i}" for i in range(n_topic)],
                len(self.dataset.getFeaturesAs("array")),
            )
        )
        fakeTopicDataset = PandasDataset(
            features=self.dataset.getFeaturesAs("pandas"), labels=newLabels
        )
        benchmarkMetrics = CoherenceMetric(
            dataset=fakeTopicDataset,
            numberOfDoc=self.numberOfDoc,
            dictPurifiedKey=self.dictPurifiedKey,
            mapKeyDoc=self.mapKeyDoc,
            reductionParams=self.reductionParams,
            segmentationMeasures=self.segmentationMeasures,
        )
        union_summary = pd.concat(
            [benchmarkMetrics.summary, self.summary],
            keys=["Benchmarck Topic", "Original Topic"],
        )
        dataToPlot = union_summary.rename_axis(
            ["Topic Type", "Topic name", "Partition", "Measure"]
        ).reset_index()
        g = relplot(
            x="totalDocNumber",
            y=y,
            col="Partition",
            row="Measure",
            hue="Topic Type",
            kind="scatter",
            data=dataToPlot,
            alpha=0.5,
            size="nUsedKeys",
            sizes=(40, 500),
            facet_kws={"sharey": False, "sharex": False},
        )
        g.set(xscale="log")
