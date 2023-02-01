"""
General functions.

Probably these functions are not used by any other module anymore (to be deprecated?).
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from py4ai.core.logging import getDefaultLogger
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

from py4ai.analytics.ml import ArrayLike


logger = getDefaultLogger()
CorrFunc = Callable[[ArrayLike, ArrayLike], float]


def cutOffWithPCA(
    df: ArrayLike, cutOff: float = 0.95, initial_ncomp: int = 8, step: int = 10
) -> Tuple[PCA, pd.DataFrame]:
    """
    Recursive PCA with cut off on variance.

    Applies recursive PCA up to an explained variance equal to a given cut off.

    :param df: input dataframe to be dimensionally reduced
    :param cutOff: target explained variance. must be less than 1
    :param initial_ncomp: number of initial principal components
    :param step: number of principal comp to add at each step

    :raises ValueError: if number of initial components equal or greater than n_col of input df, if cut-off is <= 0 or if cut-off is > 0

    :return: a tuple with the PCA object and a pd dataframe with PCs
    """
    log = logger()

    ncomp = initial_ncomp
    if initial_ncomp > df.shape[1]:
        raise ValueError("Cannot run PCA. Number of initial components greater than number of columns")

    if cutOff > 1:
        raise ValueError("The cutOff must be <=> 1.")

    if cutOff <= 0:
        raise ValueError("Cut-off must be > 0")

    def _pca_cumulative_expl_var(df: pd.DataFrame, ncomp: int):
        pca = PCA(n_components=min(ncomp, df.shape[1]))
        _ = pca.fit(df)
        return pca, pca.explained_variance_ratio_.cumsum()

    pca, cumulative_expl_var = _pca_cumulative_expl_var(df, ncomp)

    while (cumulative_expl_var[-1] < cutOff) and (ncomp < min(df.shape)):
        ncomp += step
        log.debug("Testing with %d components" % ncomp)
        pca, cumulative_expl_var = _pca_cumulative_expl_var(df, ncomp)
        log.debug("Explained Var: %2.3e" % cumulative_expl_var)

    n_components = len(cumulative_expl_var[cumulative_expl_var < cutOff]) + 1
    pca, _ = _pca_cumulative_expl_var(df, n_components)
    return pca, pd.DataFrame(pca.transform(df), index=df.index)


def computeCorrelation(
    x: pd.DataFrame,
    y: Optional[pd.DataFrame] = None,
    method: Union[CorrFunc, str] = 'pearson',
) -> pd.DataFrame:
    """
    Compute (auto/pairwise)correlation.

    Computes (auto/pairwise)correlation given a metrics specified by the user.

    :param x: a  dataframe to (auto)correlate
    :param y: an optional dataframe to be correlated to X. If missing the auto-correlation is performed
    :param method: {‘pearson’, ‘kendall’, ‘spearman’} or callable. Method of correlation:
                        - pearson : standard correlation coefficient
                        - kendall : Kendall Tau correlation coefficient
                        - spearman : Spearman rank correlation
                        - callable: callable with input two 1d ndarrays like and returning a float. Note that the returned matrix from corr will have 1 along the diagonals and will be symmetric regardless of the callable’s behavior.

    :return: a pd dataframe with the output of the correlation
    """
    return x.corr(method=method) if y is None else pd.DataFrame({k: x.corrwith(y[k], method=method) for k in y.columns})


def crosscorr(
    x: pd.Series, y: pd.Series, lags: int = 0, pvalflag: bool = False
) -> pd.DataFrame:
    """
    Lag-N cross correlation.

    :param x: first object to correlate
    :param y: second object with the same lenght of x
    :param lags: l int, default 0
    :param pvalflag: a flag to compute also pvalues from pearson correlation tests

    :return: a pd.DataFrame
    """
    if pvalflag:
        rho = []
        for lag in range(lags + 1):
            rho.append(
                pearsonr(
                    x[~y.shift(lag).isnull()], y.shift(lag)[~y.shift(lag).isnull()]
                )
            )

        corrs = [x[0] for x in rho]
        pvals = [x[1] for x in rho]
        return pd.DataFrame({"Corr": corrs, "Pval": pvals})
    else:
        rho = []
        for lag in range(lags + 1):
            rho.append(x.corr(y.shift(lag)))
        return pd.DataFrame({"Corr": rho})


def mergeDataFrame(d: Dict[str, pd.DataFrame], join: str = "outer") -> pd.DataFrame:
    """
    Merge a dictionary of pandas dataframes.

    :param d: a dictionary of dataframes
    :param join: {'inner', 'outer'}, default 'outer'
        How to handle indexes on other axis (or axes).

    :return: a merged dataframe
    """
    dfs = []
    for k, v in d.items():
        df = v.copy()
        df.columns = list(map(lambda x: k + str(x), v.columns))
        dfs.append(df)
    return pd.concat(dfs, join=join, axis=1)


def filterColumnsWith(df: pd.DataFrame, prefix: Union[str, List[str]]) -> pd.DataFrame:
    """
    Select columns containing values.

    Given a pd.DataFrame and name or a list of strings select the columns whose name contain them as substings.

    :param df: input dataframe
    :param prefix: string or list of strings

    :return: a dataframe containing the selected columns
    """
    _prefix = [el for el in prefix] if isinstance(prefix, list) else [prefix]
    return df[[col for col in df.columns if any([pre in col for pre in _prefix])]]


def flattenMultiIndex(
    df: pd.DataFrame,
    axis: int = 1,
    sep: Optional[str] = None
) -> pd.DataFrame:
    """
    Flatten the multi index to a single level.

    :param df: a multi indexed dataframe
    :param axis: {0, 1} which axis
    :param sep: Optional. Specifies the separator to use when collapsing the multi index. If None a 1-d index of tuple will be produced

    :return: a pd dataframe with single indexing
    """
    _df = df.copy()
    if axis == 1:
        _df.columns = [col for col in _df.columns.values] if sep is None else [sep.join(col) for col in _df.columns.values]
    else:
        _df.index = [col for col in _df.index.values] if sep is None else [sep.join(col) for col in _df.index.values]
    return _df


def generate_multi_index(
    df: pd.DataFrame,
    axis: int = 1,
    sep: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate a multi-index from a single level index dataframe.

    :param df: a multi indexed dataframe
    :param axis: {0, 1} which axis
    :param sep: Optional. If a string the multi-index will be calculated splitting the strings by sep. If None the Multi-Index will be calculated by tuples.
    :raises ValueError: if dataframe is already multi-index
    :return: a pd dataframe with single indexing
    """
    if df.axes[axis].nlevels > 1:
        raise ValueError(" DataFrame is multi-index along axis %d" % axis)

    out = df.copy()

    if axis == 1:
        out.columns = pd.MultiIndex.from_tuples(
            [tuple(c.split(sep)) for c in out.columns]
        ) if sep is not None else pd.MultiIndex.from_tuples(
            [c if isinstance(c, tuple) else (c,) for c in out.columns]
        )
    elif axis == 0:
        out.index = pd.MultiIndex.from_tuples(
            [tuple(c.split(sep)) for c in out.index]
        ) if sep is not None else pd.MultiIndex.from_tuples(
            [c if isinstance(c, tuple) else (c,) for c in out.index]
        )
    return out
