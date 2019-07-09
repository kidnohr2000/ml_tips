# -*- coding: utf-8 -*-
# vim:tabstop=4:shiftwidth=4:expandtab

from copy import deepcopy as copy

import os
import re
import math
import gc

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import scipy.stats as st

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .converter import standard_scaler
from .utils.pd_utils import isinstance_pd

pd.set_option('display.max_columns', 200)
# カラム内の文字数
pd.set_option("display.max_colwidth", 200)
#行数
pd.set_option("display.max_rows", 200)

# matplotlib の日本語表示
rcParams['font.family'] = 'IPAPGothic'


def get_missing_data(df):
    list_na = (df.isnull().sum() / len(df)) * 100
    list_na = list_na.sort_values(ascending=False)
    return pd.DataFrame({'Missing Ratio' :list_na})


def null_ratio_visualizer(df, num=None):

    if num:
        df = df[:num]
    missing_data = get_missing_data(df)
    display(missing_data)

    plt.figure(figsize=(15, 6))

    plt.xticks(rotation='90')
    sns.barplot(x=missing_data.index, y=missing_data['Missing Ratio'])

    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)

    plt.show()


def log_convert_vidualizer(df, sel_col):

    _, a = plt.subplots(nrows=len(sel_col), ncols=2, figsize=(14, 2 * len(sel_col)))

    a = a.ravel()
    for idx,ax in tqdm(enumerate(zip(a[::2], a[1::2]))):
        l = sel_col[idx]
        ax1, ax2 = ax
        df[l].hist(bins=50, ax=ax1)
        ax1.set_xlabel(l)
        ax1.set_label('count')
        ax1.set_title('original')
        df[l].apply(np.log1p).hist(bins=50, ax=ax2)
        ax2.set_xlabel(l)
        ax2.set_label('count')
        ax2.set_title('log convert')
    plt.tight_layout()
    plt.show()


def decompose_model_visualizer(
    X, y, model=PCA(), visual_dim=[0, 1], ex_outlier=False
):

    if isinstance_pd(X):
        X = X.values
    if isinstance_pd(y):
        y = y.values

    X_std, _, _ = standard_scaler(X)
    if ex_outlier:
        X_std[np.where(X < -3)] = -3
        X_std[np.where(X > 3)] = 3

    X_tr = model.fit_transform(X_std)

    X_p1 = X_tr[np.where(y==1), visual_dim[0]]
    X_n1 = X_tr[np.where(y==0), visual_dim[0]]
    X_p2 = X_tr[np.where(y==1), visual_dim[1]]
    X_n2 = X_tr[np.where(y==0), visual_dim[1]]

    plt.figure(figsize=(8, 6))

    plt.scatter(X_p1, X_p2, color='g', label='Positive', alpha=0.5)
    plt.scatter(X_n1, X_n2, color='r', label='Negative', alpha=0.5)

    plt.xlabel('PC{}'.format(visual_dim[0]))
    plt.ylabel('PC{}'.format(visual_dim[1]))

    plt.title(model.__class__.__name__)

    plt.legend()
    plt.tight_layout()

    plt.show()


def create_corrmap(df, ex_col):
    corrmat = df.drop(columns=ex_col).corr()
    plt.subplots(figsize=(12,9))
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.show()


def histgram_per_class(X, y, title='histgram_per_class'):
    if isinstance_pd(X):
        cols = X.columns
        X = X.values
    else:
        cols = ['col_{i}'.format(i) for i in range(X.shape[1])]
    if isinstance_pd(X):
        y = y.values

    line_num = math.ceil(len(cols)/2)

    _,a = plt.subplots(nrows=line_num, ncols=2, figsize=(8, 2 * line_num))
    a = a.ravel()
    for idx,ax in enumerate(a):
        if idx == len(cols):
            break
        col = cols[idx]
        xi = X[:, idx]

        x_idx = xi != np.nan

        sns.distplot(
            xi[(x_idx) & (y==1)], kde=True, bins=20, color='red', ax=ax,
        )
        sns.distplot(
            xi[(x_idx) & (y==0)], kde=True, bins=20, color='blue', ax=ax,
        )
        ax.set_xlabel(col)
        ax.set_ylabel('probability density')
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def importance_barplot(X, y, model, upper=30):

    # 例
    # params = dict(
    #     n_estimators=50,
    # )
    # model = RandomForestRegressor(**params)

    res_dic = {}

    model.fit(X, y)

    importances = model.feature_importances_

    ## argsort はsortした index をとってくる
    indices = np.argsort(importances)[::-1][:upper]

    res_dic['importance_columns'] = np.array(X.columns[indices])
    res_dic['importance_columns_score'] = importances[indices]

    plt.figure(figsize=(10,6))
    sns.set()
    plt.title("Feature Importances: {}".format(model.__class__.__name__))
    plt.bar(range(len(importances[indices])), importances[indices], align='center')
    plt.xticks(range(len(importances[indices])), X.columns[indices], rotation=90)
    plt.xlim([-1, len(importances[indices])])

    plt.tight_layout()
    plt.show()

    return res_dic