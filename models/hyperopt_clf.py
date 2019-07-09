# -*- coding: utf-8 -*-
# vim:tabstop=4:shiftwidth=4:expandtab

# 複数のモデルの計算結果の傾向より、分類を予測するメタモデル

import functools
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK, space_eval
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

RANDOM_STATE = 123
MP = 4

HPO_PARAMS = {
    'SVC': {
        'C': hp.loguniform('C', -8, 2),
        'gamma': hp.loguniform('gamma', -8, 2),
        'kernel': hp.choice('kernel', ['rbf', 'poly', 'sigmoid']),
        'random_state': RANDOM_STATE
    },
    'XGB': {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        # Increase this number if you have more cores. Otherwise, remove it and it will default
        # to the maxium number.
        'n_jobs': MP,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1,
        'seed': RANDOM_STATE
    },
    'LR': {
        'C': hp.loguniform('C', 0.00001, 100),
        'penalty': hp.choice('penalty', ['l1', 'l2']),
        'random_state': RANDOM_STATE
    },
    'RF': {
        'max_depth': hp.choice('max_depth', range(1,20)),
        # 'max_features': hp.choice('max_features', range(1,150)),
        'n_estimators': hp.choice('n_estimators', range(100,500)),
        'criterion': hp.choice('criterion', ["gini", "entropy"])
    },
    'KN': {
        'n_neighbors': hp.choice('n_neighbors', range(4, 9)),
        'p': hp.choice('p', [1, 2]),
        'metric': 'minkowski',
        'n_jobs': MP,
    },
    'MLP': {
        'solver': 'lbfgs',
        'alpha': hp.loguniform('alpha', 10**-5, 10**-1),
        'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(i + 2, 2) for i in range(3)]),
        'random_state': RANDOM_STATE,
    }
}

CLF_DICT = {
    'SVC': SVC,
    'XGB': xgb,
    'LR': LogisticRegression,
    'RF': RandomForestClassifier,
    'KN': KNeighborsClassifier,
    'MLP': MLPClassifier,
}

def trail_run(objective, hyperopt_parameters, max_evals=200):
    # 試行の過程を記録するインスタンス
    trials = Trials()

    best = fmin(
        # 最小化する値を定義した関数
        objective,
        # 探索するパラメータのdictもしくはlist
        hyperopt_parameters,
        # どのロジックを利用するか、基本的にはtpe.suggestでok
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        # 試行の過程を出力
        verbose=1
    )

    return space_eval(hyperopt_parameters, best), trials

# 例:
# for k in HPO_PARAMS.keys():
#     chs = Clf_HpoSearch(data, target[:, 0], k)
#     best, trials = trail_run(chs.objective, chs.hyperopt_parameters, max_evals=5)
#     print(best)

def f1_wrapper(func, average=None):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        kwargs.update(average=average)
        return func(*args, **kwargs)
    return inner


METRIC_DICT = {
    'auc': roc_auc_score,
    'f1': f1_wrapper(f1_score, average='micro')
}


class Clf_HpoSearch(object):
    def __init__(self, X, y, model_name, metric_name='auc'):
        self.X, self.y = X, y
        self.suffle_data()
        self.model_name = model_name
        self.hyperopt_parameters = HPO_PARAMS[model_name]
        self.metric = METRIC_DICT[metric_name]
        # self.objective = getattr(self, "{}_objective".format(model_name.lower()))

    def suffle_data(self, test_size=0.2, random_state=None):
        x_train, x_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size,
            # random_state=random_state
        )
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def objective(self, args):
        if self.model_name == 'XGB':
            return self.xgb_objective(args)
        else:
            return self.clf_objective(args)

    def xgb_objective(self, args):
        print("Training with params: ")
        print(args)
        num_round = int(args['n_estimators'])
        del args['n_estimators']

        dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
        dvalid = xgb.DMatrix(self.x_test, label=self.y_test)
        # watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
        gbm_model = xgb.train(
            args,
            dtrain,
            num_round,
            # evals=watchlist,
            verbose_eval=True
        )
        pred = gbm_model.predict(
            dvalid,
            ntree_limit=gbm_model.best_iteration + 1
        )
        score = self.metric(self.y_test, pred)
        # TODO: Add the importance for the selected features
        print("\t{0} {1}\n\n".format(self.metric.__qualname__, score))
        # The score function should return the loss (1-score)
        # since the optimize function looks for the minimum
        loss = 1 - score
        return {'loss': loss, 'status': STATUS_OK}

    def clf_objective(self, args):
        print("Training with params: ")
        print(args)
        clf = CLF_DICT[self.model_name](**args)
        clf.fit(self.x_train, self.y_train)
        print(self.x_train.shape, self.y_train.shape)
        # validationデータを使用して、ラベルの予測
        pred = clf.predict(self.x_test)
        # 予測ラベルと正解ラベルを使用してmicro f1を計算
        score = self.metric(self.y_test, pred)
        # 今回はmicro f1を最大化したいので、-1をかけて最小化に合わせる
        print("\t{0} {1}\n\n".format(self.metric.__qualname__, score))
        return {'loss': -1 * score, 'status': STATUS_OK}
