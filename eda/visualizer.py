# -*- coding: utf-8 -*-
# vim:tabstop=4:shiftwidth=4:expandtab

from copy import deepcopy as copy

import os
import re
import gc

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .converter import standard_scaler

pd.set_option('display.max_columns', 200)
# カラム内の文字数
pd.set_option("display.max_colwidth", 200)
#行数
pd.set_option("display.max_rows", 200)


def decompose_model_visualize(
    X, y, model=PCA(), visual_dim=[0, 1], ex_outlier=False
):

    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(X, pd.DataFrame):
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

    plt.xlabel('dimention {}'.format(visual_dim[0]))
    plt.ylabel('dimention {}'.format(visual_dim[1]))

    plt.title(model.__class__.__name__)

    plt.legend()
    plt.tight_layout()

    plt.show()