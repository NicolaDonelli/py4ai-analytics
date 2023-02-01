"""Functions to plot bar-charts using matplotlib."""
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def binomial_proportion_confint(
    size: pd.Series, success: pd.Series, z: float = 1.96
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Confidence interval for binomial proportion.

    :param size: pd.Series with number of trials
    :param success: pd.Series with number of success
    :param z: 1 -alpha/2 quantile of a standard notmal distribution
    :return: tuple of success / size, lower bound and upper bound
    """
    p_ = success / size

    first_term = (success.add(z**2 / 2)) / (size.add(z**2))
    second_term = (
        z / (size.add(z**2)) * np.sqrt(success * (size - success) / size + z**2 / 4)
    )

    low = first_term - second_term
    upb = first_term + second_term

    return p_, low, upb


def plot_bivariate(
    df: pd.Series,
    y: pd.Series,
    figsize: Optional[tuple] = None,
    y_labels: Optional[List[str]] = None,
    min_proportion: float = 0.01,
) -> None:
    """
    Bar plot for categorical feature versus binary target.

    :param df: categorical feature
    :param y: target
    :param figsize: figsize. Default None
    :param y_labels: lables for dual y axis. Default ['Categories size', 'Target']
    :param min_proportion: minimum proportion to represent categorical levels.
        If one or more levels are too small, they are grouped under a new level "other"
    """
    y_labels = ["Categories size", "Target"] if y_labels is None else y_labels
    x = pd.DataFrame({"category": df.astype(str), "target": y})
    r = (
        x.groupby("category")
        .agg(size=("target", "size"), success=("target", "sum"))
        .reset_index()
    )
    r.loc[r["size"] <= min_proportion * r["size"].sum(), "category"] = "other"
    r = r.groupby("category").agg("sum").reset_index()
    p_, l_, u_ = binomial_proportion_confint(r["size"], r["success"])
    order = (-np.array(r["size"])).argsort()

    fig, ax0 = plt.subplots(figsize=figsize)
    plt.xticks(rotation=90)

    plt.xlabel(df.name)
    ax0.bar(np.array(r["category"])[order], np.array(r["size"])[order], color="blue")
    ax0.set_ylabel(y_labels[0], labelpad=20)
    ax1 = ax0.twinx()
    ax1.set_ylabel(y_labels[1], labelpad=20)

    plt.hlines(y.mean(), -0.5, p_.shape[0] - 0.5, color="red", linestyle=":", alpha=0.8)

    ax1.plot(p_.index, p_.loc[p_.index][order], color="red", alpha=0.5)
    ax1.scatter(p_.index, l_.loc[p_.index[order]], color="red", alpha=0.5)
    ax1.scatter(p_.index, u_.loc[p_.index[order]], color="red", alpha=0.5)
    ax1.scatter(p_.index, p_.loc[p_.index[order]], color="red")


def plot_bivariate_num(
    df: pd.Series,
    y: pd.Series,
    figsize: Optional[tuple] = None,
    y_labels: Optional[List[str]] = None,
    n_bins: int = 10,
) -> None:
    """
    Bar plot for numerical feature versus binary target.

    :param df: numeric feature
    :param y: target
    :param figsize: figsize. Default None
    :param y_labels: lables for dual y axis. Default ['Bin size', 'Target']
    :param n_bins: number of bins of histogram
    """
    y_labels = ["Bin size", "Target"] if y_labels is None else y_labels
    fig, ax0 = plt.subplots(figsize=figsize)
    plt.xticks(rotation=90)
    plt.xlabel(df.name)
    (n, bins, patches) = ax0.hist(df, color="blue", bins=n_bins)
    x = pd.DataFrame({"bin": pd.cut(df, bins=bins), "target": y})
    x["middle_bin"] = x["bin"].apply(lambda x: x.mid)
    r = (
        x.groupby("middle_bin")
        .agg(size=("target", "size"), success=("target", "sum"))
        .reset_index()
    )
    p_, l_, u_ = binomial_proportion_confint(r["size"], r["success"])
    ax1 = ax0.twinx()
    ax0.set_ylabel(y_labels[0], labelpad=20)
    ax1.set_ylabel(y_labels[1], labelpad=20)
    bin_size = r["middle_bin"][1] - r["middle_bin"][0]
    plt.hlines(
        y.mean(),
        min(r["middle_bin"]) - bin_size,
        max(r["middle_bin"]) + bin_size,
        color="red",
        linestyle=":",
        alpha=0.8,
    )
    ax1.plot(r["middle_bin"].to_list(), p_.loc[p_.index], color="red", alpha=0.5)
    ax1.scatter(r["middle_bin"], l_.loc[p_.index], color="red", alpha=0.5)
    ax1.scatter(r["middle_bin"], u_.loc[p_.index], color="red", alpha=0.5)
    ax1.scatter(r["middle_bin"], p_.loc[p_.index], color="red")
