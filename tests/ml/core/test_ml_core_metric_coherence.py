import unittest
from collections import namedtuple
from itertools import product

import numpy as np
from py4ai.core.tests.core import TestCase, logTest
from gensim.topic_coherence import (
    direct_confirmation_measure,
    indirect_confirmation_measure,
    text_analysis,
)

from py4ai.analytics.ml.core.estimator.selector import DeleteConstantFeatures
from py4ai.analytics.ml.core.metric.coherence import (
    CoherenceEvaluator,
    CoherenceMetric,
    ReductionMethod,
    jaccardIndex,
    segmentationsFuns,
)
from py4ai.analytics.ml.core.pipeline.transformer import PipelineEstimator
from py4ai.analytics.ml.core.pipeline.transformer import PipelineTransformer
from tests.helpers import FakeTrasformer, allSegmentationMeasureSet
from tests.resources.metric_coherence_data import docs, topicDataset, dictMapKeyDoc


class TestMeasureCoherence(TestCase):
    topicEvaluator: CoherenceEvaluator
    fakePipelineTrasformer: PipelineTransformer
    fakeEvaluator: CoherenceEvaluator

    @classmethod
    def setUpClass(cls) -> None:
        cls.topicEvaluator = CoherenceEvaluator(
            docs, segmentationMeasures=allSegmentationMeasureSet
        )

        cls.fakePipelineTrasformer = PipelineTransformer(
            steps=[("Step1", FakeTrasformer())],
            estimator=PipelineEstimator(steps=[("a", DeleteConstantFeatures())]),
        )
        cls.fakeEvaluator = CoherenceEvaluator(
            docs, textTransformer=cls.fakePipelineTrasformer
        )

    @logTest
    def testEvaluator(self) -> None:
        result = self.topicEvaluator.evaluate(topicDataset)
        self.assertIsInstance(result, CoherenceMetric)
        self.assertEqual(result.mapKeyDoc, dictMapKeyDoc)

    @logTest
    def testPipelineTransformer(self) -> None:
        self.assertTrue(
            all(
                [
                    text == "textpurified"
                    for text in self.fakeEvaluator.corpusProcessed.getFeaturesAs(
                        "array"
                    ).flat
                ]
            )
        )
        result = self.fakeEvaluator.evaluate(topicDataset)
        self.assertIsInstance(result, CoherenceMetric)
        self.assertTrue(
            all(
                [
                    len(value) == len(self.fakeEvaluator.corpusProcessed)
                    for key, value in result.mapKeyDoc.items()
                ]
            )
        )

    @logTest
    def testMeasuresAreRet(self) -> None:
        metric = self.topicEvaluator.evaluate(topicDataset)
        self.assertIsInstance(metric.marginalMeasures, dict)

    @logTest
    def testMeasuresGensim(self) -> None:
        metric = self.topicEvaluator.evaluate(topicDataset)
        for topic, value in metric.marginalMeasures.items():
            ll_keys = list(metric.reducedKeys[topic])
            id2token = {i: ll_keys[i] for i in range(len(ll_keys))}
            token2id = {v: k for k, v in id2token.items()}
            dictionary = namedtuple("Dictionary", "token2id, id2token")(
                token2id, id2token
            )
            accumulator = text_analysis.InvertedIndexAccumulator(
                set(id2token.keys()), dictionary
            )
            accumulator._inverted_index = {
                accumulator.id2contiguous[token2id[tkey]]: metric.unionWordsDocs(tkey)
                for tkey in ll_keys
            }
            accumulator._num_docs = self.topicEvaluator.lenCorpus

            for mysegm_name, value2 in value.items():
                mysegm = value2["segmentation"]
                for measure, value3 in value2.items():
                    to_test = False
                    gensim_m = None
                    if (measure in ["m_lr", "m_nlr"]) & (mysegm_name == "S_one_one"):
                        to_test = True
                        gensim_segm = [
                            [(token2id[S[0][0]], token2id[S[1][0]])] for S in mysegm
                        ]

                        if measure == "m_lr":
                            gensim_m = direct_confirmation_measure.log_ratio_measure(
                                gensim_segm, accumulator
                            )
                        if measure == "m_nlr":
                            gensim_m = direct_confirmation_measure.log_ratio_measure(
                                gensim_segm, accumulator, normalize=True
                            )

                    if measure == "m_cos_nlr":
                        to_test = True
                        gensim_segm = [
                            [
                                (
                                    token2id[S[0][0]],
                                    np.array(
                                        [token2id[S[1][j]] for j in range(len(S[1]))]
                                    ),
                                )
                            ]
                            for S in mysegm
                        ]

                        keywords_costest = np.array(
                            [list(id2token.keys()) for s in gensim_segm]
                        )
                        gensim_m = indirect_confirmation_measure.cosine_similarity(
                            gensim_segm, accumulator, np.array(keywords_costest)
                        )

                    if to_test:
                        self.logger.debug(f"TRY: {topic}, {mysegm_name}, {measure}")
                        self.assertAlmostEqual(
                            sum(np.array(value3) - np.array(gensim_m)),
                            0,
                            msg=f"Measure {measure}",
                        )

                    if measure == "m_d":
                        self.logger.debug(
                            f"TRY: {topic}, {mysegm_name}, {measure}  -  only range"
                        )
                        self.assertTrue(all(map(lambda x: x <= 1, value3)))
                        self.assertTrue(all(map(lambda x: x >= -1, value3)))

    @logTest
    def testSegmentation(self) -> None:
        setElement = {"a", "b", "c", "d"}

        self.logger.info("Control One One and One Pre segm ")
        segm_one_one_real = [
            ([e1], [e2]) for e1 in setElement for e2 in setElement if e1 != e2
        ]
        segm_one_one = segmentationsFuns["S_one_one"](list(setElement))
        self.assertEqual(len(segm_one_one), len(segm_one_one_real))
        for tel in segm_one_one_real:
            self.assertIn(tel, segm_one_one)
        segm_one_pre = segmentationsFuns["S_one_pre"](list(setElement))
        self.assertEqual(len(segm_one_pre), len(segm_one_one) / 2)

        self.logger.info("Control One Any segm ")
        segm_one_any = segmentationsFuns["S_one_any"](list(setElement))
        self.assertEqual(
            len(segm_one_any), (len(setElement) * (pow(2, len(setElement) - 1) - 1))
        )
        self.assertEqual(len({tel[0][0] for tel in segm_one_any}), len(setElement))
        for tel in segm_one_any:
            self.assertEqual(len(tel[0]), 1)
            self.assertNotIn(tel[0][0], tel[1])

        self.logger.info("Control One Set segm")
        segm_one_set = segmentationsFuns["S_one_set"](list(setElement))
        self.assertEqual(len(segm_one_set), len(setElement))
        for tel in segm_one_set:
            self.assertEqual(len(tel[0]), 1)
            self.assertEqual(len(set(tel[1])), len(setElement))

        self.logger.info("Control One All segm")
        segm_one_all = segmentationsFuns["S_one_all"](list(setElement))
        self.assertEqual(len(segm_one_all), len(setElement))
        for tel in segm_one_all:
            self.assertEqual(len(tel[0]), 1)
            self.assertEqual(len(set(tel[1])), (len(setElement) - 1))
            self.assertNotIn(tel[0][0], tel[1])

    @logTest
    def testKeyReduction(self) -> None:
        thr = 0.33
        ntop = 2
        rdm = ReductionMethod(True, thr, ntop)
        oldmetric = self.topicEvaluator.evaluate(topicDataset)
        metric = oldmetric.metricWithNewParams(reductionParams=rdm)
        self.logger.info("Test Reduction")
        for _, keywords in metric.reducedKeys.items():
            self.assertLessEqual(len(keywords), ntop)
            for tkey in keywords:
                if len(tkey) > 1:
                    for word1, word2 in product(tkey, tkey):
                        set1 = set(word1.split(" "))
                        set2 = set(word2.split(" "))
                        self.assertGreater(jaccardIndex(set(set1), set(set2)), thr)
        # test all keywords grouped
        thr = -0.1
        ntop = None
        rdm = ReductionMethod(True, thr, ntop)
        metric = oldmetric.metricWithNewParams(reductionParams=rdm)
        for _, keywords in metric.reducedKeys.items():
            self.assertEqual(len(keywords), 1)

    @logTest
    def testJaccardIndex(self) -> None:
        idx = jaccardIndex({"a", "b", "c"}, {"a", "b", "d", "e"})
        self.logger.info("Test jaccard index")
        self.assertAlmostEqual(idx, 2 / 5)


if __name__ == "__main__":
    unittest.main()
