# -*- coding: utf-8 -*-
# vim:tabstop=4:shiftwidth=4:expandtab

import os
import re
import math
import gc
from inspect import signature
from copy import deepcopy as copy

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

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

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


def get_col_high_corr(df, thres=0.95, ex_cols=[], _abs=True):
    _corrmat = df.drop(columns=ex_cols).corr()
    if _abs:
        corrmat = _corrmat.applymap(np.abs)
    else:
        corrmat = copy(_corrmat)
    indices = np.where(corrmat > thres)
    indices = [(corrmat.index[x], corrmat.columns[y], round(_corrmat.iloc[x, y], 3)) for x, y in zip(*indices)
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

    plt.legend()
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


# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument

def xbtraincv_plot_roc_prerec(X, y, params={}, target_name=None):
    if target_name is not None:
        print(f'target: {target_name}')
    sns.set(font='IPAPGothic')
    _, a = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))
    a = a.ravel()
    ax1 = a[0]
    ax2 = a[1]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

#     from IPython.core.debugger import Pdb; Pdb().set_trace()

    if isinstance_pd(X):
        X = X.values
    if isinstance_pd(y):
        y = y.values

    rocs = []
    prs = []
    aucs = []
    average_precisions = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        x_train = X[train_index, :]
        y_train = y[train_index]
        x_test = X[test_index, :]
        y_test = y[test_index]

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dvalid = xgb.DMatrix(x_test, label=y_test)

        num_round = int(params['n_estimators'])

        gbm_model = xgb.train(
            params,
            dtrain,
            num_round,
            # evals=watchlist,
            # verbose_eval=True
        )
        pred_values = gbm_model.predict(
            dvalid,
            ntree_limit=gbm_model.best_iteration + 1
        )

        fpr, tpr, _ = roc_curve(y_test, pred_values)
        precision, recall, _ = precision_recall_curve(y_test, pred_values)

        rocs.append((fpr, tpr))
        prs.append((precision, recall))

        roc_auc = roc_auc_score(y_test, pred_values)
        aucs.append(roc_auc)

        aps = average_precision_score(y_test, pred_values)
        average_precisions.append(aps)

        ax1.plot(fpr, tpr, lw=1, alpha=1,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        ax2.plot(recall, precision, lw=1, alpha=1,
             label='AP fold %d (AUC = %0.2f)' % (i, aps))

#     from IPython.core.debugger import Pdb; Pdb().set_trace()
#     fprs, tprs = zip(*rocs)
#     mean_fpr = np.mean(fprs, axis=0)
#     mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

#     ax1.plot(mean_fpr, mean_tpr, color='b',
#          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#          lw=2, alpha=.8)

#     precisions, recalls = zip(*prs)
#     mean_precision = np.mean(precisions, axis=0)
#     mean_recall = np.mean(recall, axis=0)
    mean_average_precisions = np.mean(average_precisions)
    std_average_precisions = np.std(average_precisions)

#     ax2.plot(mean_recall, mean_precision, color='b',
#          label=r'Mean AP ( %0.2f $\pm$ %0.2f)' % (mean_average_precisions, std_average_precisions),
#          lw=2, alpha=.8)

#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                     label=r'$\pm$ 1 std. dev.')

#     std_pre_recs = np.std(precisions, axis=0)
#     pre_pre_upper = np.minimum(mean_pre_recs + std_pre_recs, 1)
#     pre_pre_lower = np.maximum(mean_pre_recs - std_pre_recs, 0)
#     ax2.fill_between(mean_recall, pre_pre_lower, pre_pre_upper, color='grey', alpha=.2,
#                     label=r'$\pm$ 1 std. dev.')

    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver operating characteristic: AUC={0:0.2f} $\pm$ {1:0.2f}'.format(mean_auc, std_auc))
    ax1.legend(loc="lower right")

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_title('Precision-Recall curve: AUC={0:0.2f} $\pm$ {1:0.2f}'.format(
              mean_average_precisions, std_average_precisions))
    ax2.legend(loc="lower right")
    plt.show()


def plot_roc_prerec(test, score, target_name=None):
    if target_name is not None:
        print(target_name)
    roc_score = roc_auc_score(test, score)
    print('ROC score: {0:0.2f}'.format(
          roc_score))
    average_precision = average_precision_score(test, score)
    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))

    sns.set(font='IPAPGothic')
    _, a = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))
    a = a.ravel()
    ax1 = a[0]
    ax2 = a[1]

    # ROC
    fpr, tpr, _ = roc_curve(test, score)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    ax1.step(fpr, tpr, lw=2, color='b', alpha=1,
             where='post')
    ax1.fill_between(fpr, tpr, alpha=0.2, color='b', **step_kwargs)

    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver operating characteristic: 'ROC score: {0:0.2f}'.format(
          roc_score))
    ax1.legend(loc="lower right")

    # PRE-REC
    precision, recall, _ = precision_recall_curve(test, score)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    ax2.step(recall, precision, color='b', alpha=1,
             where='post')
    ax2.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlim([0.0, 1.05])
    ax2.set_title('Precision-Recall curve: AUC (AP)={0:0.2f}'.format(
              average_precision))
    ax2.legend(loc="lower right")
    plt.show()


def plotConfusionMatrix(y_test,pred,y_test_legit,y_test_fraud):

    cfn_matrix = confusion_matrix(y_test,pred)
    cfn_norm_matrix = np.array([[1.0 / y_test_legit,1.0/y_test_legit],[1.0/y_test_fraud,1.0/y_test_fraud]])
    norm_cfn_matrix = cfn_matrix * cfn_norm_matrix

    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1,2,1)
    sns.heatmap(cfn_matrix,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')

    ax = fig.add_subplot(1,2,2)
    sns.heatmap(norm_cfn_matrix,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)

    plt.title('Normalized Confusion Matrix')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')
    plt.show()

    print('---Classification Report---')
    print(classification_report(y_test,pred))