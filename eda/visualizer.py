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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import TruncatedSVD
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


def null_ratio_visualizer(df, ex_cols=[]):

    if ex_cols:
        df = df.drop(columns=ex_cols)
    missing_data = get_missing_data(df)
    display(missing_data)

    plt.figure(figsize=(15, 6))

    plt.xticks(rotation='90')
    sns.barplot(x=missing_data.index, y=missing_data['Missing Ratio'])

    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)

    plt.show()


def null_value_check(df):
    tmp = df.isnull().sum().sort_values(ascending=False)

    if tmp.sum() == 0:
        print('no null data!!')
    else:
        display(tmp[tmp > 0])


def enpty_string_check(df):
    tmp = (df=='').sum().sort_values(ascending=False)

    if tmp.sum() == 0:
        print('no empty string data!!')
    else:
        display(tmp[tmp > 0])


def get_col_hasnull(df, thres=0):
    tmp = df.isnull().sum().sort_values(ascending=False)
    ratio = tmp / len(df)
    return tmp[ratio>thres].index.values


def get_null_rows(df, cols):
    for col in tqdm(cols):
        print('{} : null data row'.format(col))
        display(df.loc[df[col].isnull(), :].head())



def log_convert_vidualizer(df, sel_col):
    sns.set(font='IPAPGothic')
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
        X = X.values.astype(float)
    if isinstance_pd(y):
        y = y.values

    X_std, _, _ = standard_scaler(X)
    if ex_outlier:
        X_std[np.where(X < -3)] = -3
        X_std[np.where(X > 3)] = 3

    if model.__class__.__name__ == 'LinearDiscriminantAnalysis':
        X_tr = model.fit_transform(X_std, y)
    else:
        X_tr = model.fit_transform(X_std)

    X_p1 = X_tr[np.where(y==1), visual_dim[0]]
    X_n1 = X_tr[np.where(y==0), visual_dim[0]]
    X_p2 = X_tr[np.where(y==1), visual_dim[1]]
    X_n2 = X_tr[np.where(y==0), visual_dim[1]]

    plt.figure(figsize=(8, 6))

    plt.scatter(X_p1, X_p2, color='g', label='Positive', alpha=0.5)
    plt.scatter(X_n1, X_n2, color='r', label='Negative', alpha=0.5)

    plt.xlabel('dimention {}'.format(visual_dim[0]))
    plt.ylabel('dimention {}'.format(visual_dim[1]))

    plt.title(model.__class__.__name__)

    plt.legend()
    plt.tight_layout()

    plt.show()


def get_col_high_corr(df, thres=0.95, ex_cols=[]):
    corrmat = df.drop(columns=ex_cols).corr()
    indices = np.where(corrmat > thres)
    indices = [(corrmat.index[x], corrmat.columns[y]) for x, y in zip(*indices)
                                            if x != y and x < y]
    return indices


def create_corrmap(df, ex_cols=[]):
    sns.set(font='IPAPGothic')
    corrmat = df.drop(columns=ex_cols).corr()
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

    sns.set(font='IPAPGothic')
    _,a = plt.subplots(nrows=line_num, ncols=2, figsize=(12, 3 * line_num))
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


def _pairplot(df, x_vars=None, y_vars=None, hue=None):
    for col in y_vars:
        sns.set(font='IPAPGothic')
        plt.figure(figsize=(2 * len(x_vars), 10))
        g= sns.pairplot(
            df, x_vars=x_vars,  y_vars=[col],
        #     kind="reg",
            hue=hue
        )
        plt.show()


def importance_barplot(X, y, model, upper=30, train=True):

    # 例
    # params = dict(
    #     n_estimators=50,
    # )
    # model = RandomForestRegressor(**params)

    res_dic = {}
    if train:
        model.fit(X, y)

    importances = model.feature_importances_

    ## argsort はsortした index をとってくる
    indices = np.argsort(importances)[::-1][:upper]

    res_dic['importance_columns'] = np.array(X.columns[indices])
    res_dic['importance_columns_score'] = importances[indices]

    plt.figure(figsize=(10,6))
    sns.set(font='IPAPGothic')
    plt.title("Feature Importances: {}".format(model.__class__.__name__))
    plt.bar(range(len(importances[indices])), importances[indices], align='center')
    plt.xticks(range(len(importances[indices])), X.columns[indices], rotation=90)
    plt.xlim([-1, len(importances[indices])])

    plt.tight_layout()
    plt.show()

    return res_dic
