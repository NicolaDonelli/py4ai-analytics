import os
import unittest

import numpy as np
import pandas as pd
from py4ai.data.model.core import IterGenerator
from py4ai.data.model.ml import (
    LazyDataset,
    PandasDataset,
    PandasTimeIndexedDataset,
    Sample,
)
from py4ai.core.tests.core import TestCase, logTest

from py4ai.analytics.ml.core.splitter.index_based import IndexSplitter
from py4ai.analytics.ml.core.splitter.lazy_sequential import LazySequentialSplitter
from py4ai.analytics.ml.core.splitter.time_evolving import TimeEvolvingSplitter
from py4ai.analytics.ml.wrapper.sklearn.splitter import SklearnSplitter
from py4ai.analytics.ml.wrapper.sklearn.wrapper import (
    GroupKFold,
    KFold,
    StratifiedKFold,
)
from tests import DATA_FOLDER


class SplitterTest(TestCase):
    df: pd.DataFrame
    pdf: PandasDataset
    pdtids: PandasTimeIndexedDataset
    pdtidsd: PandasTimeIndexedDataset
    lazyds: LazyDataset

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.read_csv(
            os.path.join(DATA_FOLDER, "weather_nyc_short.csv"), index_col="Date"
        )
        cls.pdf = PandasDataset(
            features=cls.df.drop("TempM", axis=1), labels=cls.df.TempM
        )
        cls.pdtids = PandasTimeIndexedDataset(
            features=cls.df.drop("TempM", axis=1), labels=cls.df.TempM
        )
        cls.pdtidsd = PandasTimeIndexedDataset(
            features=cls.df.drop("TempM", axis=1), labels=(cls.df.TempM > 50).map(int)
        )

        cls.lazyds = LazyDataset(IterGenerator(cls.data_gen))

    @classmethod
    def data_gen(cls):
        for row in cls.df.iterrows():
            yield Sample(features=row[1].drop("TempM"), label=row[1].TempM, name=row[0])

    @logTest
    def test_SklearnSplitter_output_type(self) -> None:
        gkf = GroupKFold(
            partition_key=lambda x: x.date(),
            n_splits=3,
            shuffle=False,
            random_state=None,
        )
        ss = SklearnSplitter(skclass=gkf)
        train, _ = next(ss.split(self.pdtids))
        self.assertIsInstance(train, PandasTimeIndexedDataset)

    @logTest
    def test_SklearnSplitter_gfk_split(self) -> None:

        gkf = GroupKFold(
            partition_key=lambda x: x.date(),
            n_splits=3,
            shuffle=False,
            random_state=None,
        )
        ss = SklearnSplitter(skclass=gkf)
        n = sum(1 for _ in ss.split(self.pdtids))
        self.assertEqual(n, 3)

    @logTest
    def test_SklearnSplitter_kf_split(self) -> None:

        kf = KFold(n_splits=3, shuffle=False, random_state=None)
        ss = SklearnSplitter(skclass=kf)
        n = sum(1 for _ in ss.split(self.pdtids))
        self.assertEqual(n, 3)

    @logTest
    def test_SklearnSplitter_skf_split(self) -> None:

        skf = StratifiedKFold(n_splits=3, shuffle=False, random_state=None)
        ss = SklearnSplitter(skclass=skf)
        n = sum(1 for _ in ss.split(self.pdtidsd))
        self.assertEqual(n, 3)

    @logTest
    def test_TimeEvolvingSplitter_unlimited_folds(self) -> None:

        tes = TimeEvolvingSplitter(
            n_folds=np.inf,
            train_ratio=0.9,
            min_periods_per_fold=1,
            window=None,
            valid_start=None,
            g=lambda x: x,
        )

        n = sum(1 for _ in tes.split(self.pdtids))

        self.assertEqual(n, 100)

    @logTest
    def test_TimeEvolvingSplitter_k_folds(self) -> None:

        tes = TimeEvolvingSplitter(
            n_folds=7,
            train_ratio=0.9,
            min_periods_per_fold=1,
            window=None,
            valid_start=None,
            g=lambda x: x,
        )
        tes.summary(self.pdtids)
        n = sum(1 for _ in tes.split(self.pdtids))

        self.assertEqual(n, 8)

    @logTest
    def test_TimeEvolvingSplitter_2periods_per_folds(self) -> None:

        tes = TimeEvolvingSplitter(
            n_folds=np.inf,
            train_ratio=0.9,
            min_periods_per_fold=2,
            window=None,
            valid_start=None,
            g=lambda x: x,
        )

        n = sum(1 for _ in tes.split(self.pdtids))

        self.assertEqual(n, 50)

    @logTest
    def test_TimeEvolvingSplitter_valid_start(self) -> None:

        tes = TimeEvolvingSplitter(
            n_folds=np.inf,
            train_ratio=None,
            min_periods_per_fold=1,
            window=None,
            valid_start="1950-01-01",
            g=lambda x: x,
        )

        n = sum(1 for _ in tes.split(self.pdtids))

        self.assertEqual(
            n, (self.pdtids.index.date >= pd.to_datetime("1950-01-01").date()).sum()
        )

    @logTest
    def test_TimeEvolvingSplitter_windowing_with_grouping(self) -> None:

        tes = TimeEvolvingSplitter(
            n_folds=7,
            train_ratio=0.9,
            min_periods_per_fold=1,
            window=10,
            valid_start=None,
            g=lambda x: (x.year, x.week),
        )

        n = [
            len(Train.features.groupby(tes.g).groups.keys())
            for Train, _ in tes.split(self.pdtids)
        ]

        self.assertTrue((np.array(n) == 10).all())

    @logTest
    def test_LazySequentialSplitter_nfolds(self) -> None:

        lss = LazySequentialSplitter(
            initial_train_size=500, folds_size=20, n_folds=10, fixed_train_size=True
        )

        lss.summary(self.lazyds)

        n = sum(1 for _ in lss.split(self.lazyds))

        self.assertEqual(n, 10)

    @logTest
    def test_LazySequentialSplitter_fixed_train_size(self) -> None:

        lss = LazySequentialSplitter(
            initial_train_size=500, folds_size=20, n_folds=10, fixed_train_size=True
        )

        n = [len(Train.getFeaturesAs("pandas")) for Train, _ in lss.split(self.lazyds)]

        self.assertTrue((np.array(n) == 500).all())

    @logTest
    def test_LazySequentialSplitter_valid_size(self) -> None:

        lss = LazySequentialSplitter(
            initial_train_size=500, folds_size=20, n_folds=10, fixed_train_size=True
        )

        n = [len(Valid.getFeaturesAs("pandas")) for _, Valid in lss.split(self.lazyds)]

        self.assertTrue((np.array(n) == 20).all())

    @logTest
    def test_LazySequentialSplitter_varying_train(self) -> None:

        lss = LazySequentialSplitter(
            initial_train_size=500, folds_size=20, n_folds=10, fixed_train_size=False
        )

        n = [len(Train.getFeaturesAs("pandas")) for Train, _ in lss.split(self.lazyds)]

        self.assertTrue((np.diff(np.array(n)) == 20).all())

    @logTest
    def test_LazySequentialSplitter_final(self) -> None:

        lss = LazySequentialSplitter(
            initial_train_size=850, folds_size=20, n_folds=10, fixed_train_size=False
        )

        lss.summary(self.lazyds)

        n = sum(1 for _ in lss.split(self.lazyds))

        self.assertEqual(n, 8)

    @logTest
    def test_IndexSplitter(self) -> None:

        train_index = self.pdf.index[0:10]
        valid_index = self.pdf.index[20:50]
        inds = IndexSplitter(train_index, valid_index)

        train, valid = next(inds.split(self.pdf))

        self.assertEqual(len(train.index), len(train_index))
        self.assertEqual(len(valid.index), len(valid_index))


if __name__ == "__main__":
    unittest.main()
