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

pd.set_option('display.max_columns', 200)
# カラム内の文字数
pd.set_option("display.max_colwidth", 200)
#行数
pd.set_option("display.max_rows", 200)


def standard_scaler(X, eps=0):

    if len(X.shape) == 1:
        _mean = np.nanmean(X)
        _std = np.nanstd(X)
    else:
        _mean = np.nanmean(X, axis=0)
        _std = np.nanstd(X, axis=0)


    return (X - _mean) / (_std + eps), _mean, _std
