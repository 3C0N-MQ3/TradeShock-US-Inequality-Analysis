import pandas as pd
import numpy as np

# Functions for weighted aggregation


def WtSum(
    df: pd.core.frame.DataFrame,
    cols: list,
    weight_col: str,
    by_cols: list,
    outw=False,
    mask=None,
):
    """Weighted sum

    If outw=True, return the weight column as well.
    """

    out = df[[*cols, weight_col, *by_cols]].copy()
    out[[*cols, weight_col]] = out[[*cols, weight_col]].astype(
        np.float64
    )  # for sum precision

    if mask is not None:
        out = out[mask]

    for c in cols:
        out[c] = out[c] * out[weight_col]

    if outw:
        return out.groupby(by_cols)[[*cols, weight_col]].sum()
    else:
        return out.groupby(by_cols)[cols].sum()


def WtMean(
    df: pd.core.frame.DataFrame, cols: list, weight_col: str, by_cols: list, mask=None
):
    """Weighted mean"""

    out_list = []
    for c in cols:
        out = df[[c, weight_col, *by_cols]].copy()
        out[[c, weight_col]] = out[[c, weight_col]].astype(
            np.float64
        )  # for sum precision

        if mask is not None:
            out = out[mask]

        out = out[~np.isnan(out[c])]  # remove missings
        out.loc[:, c] = out.loc[:, c] * out.loc[:, weight_col]  # multiply by weights
        out = out.groupby(by_cols)[[c, weight_col]].sum()  # sum
        out.loc[:, c] = (
            out.loc[:, c] / out.loc[:, weight_col]
        )  # divide by total weights

        out_list.append(out[c])

    return pd.concat(out_list, axis=1)
