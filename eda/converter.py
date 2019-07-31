# -*- coding: utf-8 -*-
# vim:tabstop=4:shiftwidth=4:expandtab

from copy import deepcopy as copy

import os
import re
import gc
# import functools

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.max_columns', 200)
# カラム内の文字数
pd.set_option("display.max_colwidth", 200)
#行数
pd.set_option("display.max_rows", 200)


def standard_scaler(X, eps=1e-5):

    if len(X.shape) == 1:
        _mean = np.nanmean(X)
        _std = np.nanstd(X)
    else:
        _mean = np.nanmean(X, axis=0)
        _std = np.nanstd(X, axis=0)


    return (X - _mean) / (_std + eps), _mean, _std


def df_standard_scaler(df, eps=0, suffix=''):
    idxs = df.index
    if suffix:
        cols = ['{0}_{1}'.format(col, suffix) for col in df.columns]
    else:
        cols = df.columns

    X = df.values
    X_std, _mean, _std = standard_scaler(X, eps=eps)
    return pd.DataFrame(data=X_std, index=idxs, columns=cols, dtype='float'), _mean, _std


# remove col which has the same values
def rm_duplicated_columns(X):
    isdf = isinstance(X, pd.DataFrame)
    if isdf:
        df = copy(X)
    else:
        df = pd.DataFrame(X)
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index

    print('drop columns', cols_to_drop)

    if isdf:
        return df.drop(cols_to_drop, axis=1)
    else:
        return df.drop(cols_to_drop, axis=1).values


def add_static_col(_df, cols):
    df = _df.copy()
    for col in tqdm(cols):
        col_mean = '{}__mean'.format(col)
        col_std = '{}__std'.format(col)
        col_min = '{}__min'.format(col)
        col_max = '{}__max'.format(col)

        tmp = df.groupby(col)[col].agg(
            {
                col_mean: np.nanmean,
                col_std: np.nanstd,
                col_min: np.nanmin,
                col_max: np.nanmax,
            }
        ).reset_index()

        display(df)
        display(tmp)

        df = df.merge(
            tmp[[col, col_mean, col_std, col_min, col_max]],
            on=col,
            how='left',
        )
    return df

# def add_onehot_col(_df, cols, sparse=False):
#     df = _df.copy()
#     for col in tqdm(cols):
#         df_dummies = pd.get_dummies(
#             df[col], prefix=str(col) + '_', dummy_na=False,
#             sparse=sparse
#         )
#         df.drop(columns=[col], inplace=True)

#         df = df.join(df_dummies)
#     return df


def _add_onehot_col(_df, cols, sparse=False):
    df = _df.copy()
    for col in tqdm(cols):
        print('add one hot: ', col)
        unique_list = df[col].unique()
        for v in df[col].unique():
            df['{}__{}'.format(col, str(v))] = (df[col] == v) * 1

        df.drop(columns=[col], inplace=True)
    return df


def data_upsampling(df, label_name):
    labels = df[label_name].unique()
    sample_num = df[label_name].value_counts().max()

    res = None
    for l in labels:
        if res is not None:
            res = res.append(df.loc[df[label_name]==l, :].sample(sample_num, replace=True, random_state=123))
        else:
            res = df.loc[df[label_name]==l, :].sample(sample_num, replace=True, random_state=123)

    return res.sample(frac=1).reset_index(drop=True)


def data_downsampling(df, label_name):
    labels = df[label_name].unique()
    sample_num = df[label_name].value_counts().min()

    res = None
    for l in labels:
        if res is not None:
            res = res.append(df.loc[df[label_name]==l, :].sample(sample_num, random_state=123))
        else:
            res = df.loc[df[label_name]==l, :].sample(sample_num, random_state=123)

    return res.sample(frac=1).reset_index(drop=True)