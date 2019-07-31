# -*- coding: utf-8 -*-
# vim:tabstop=4:shiftwidth=4:expandtab

# 複数のモデルの計算結果の傾向より、分類を予測するメタモデル

import functools
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK, space_eval
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error,
)


RANDOM_STATE = 123
MP = 4

HPO_PARAMS = {
    'EN': {
        'alpha': hp.loguniform('alpha', -3, 2),
        'l1_ratio': hp.quniform('l1_ratio', 0, 1, 0.1),
        'tol': hp.loguniform('tol', -5, -1),
        'random_state': RANDOM_STATE,
    },
    'RID': {
        'alpha': hp.loguniform('alpha', -3, 2),
        'tol': hp.loguniform('tol', -5, -1),
        'random_state': RANDOM_STATE,
    },
    'LSS': {
        'alpha': hp.loguniform('alpha', -3, 2),
        'tol': hp.loguniform('tol', -5, -1),
        'random_state': RANDOM_STATE,
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
        'eval_metric': 'rmse',
        # 'objective': 'reg:squarederror',
        'objective': 'reg:linear',
        # Increase this number if you have more cores. Otherwise, remove it and it will default
        # to the maxium number.
        'n_jobs': MP,
        'nthread': MP,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1,
        'seed': RANDOM_STATE
    },
}

REG_DICT = {
    'EN': ElasticNet,
    'RID': Ridge,
    'EN': Lasso,
    'LSS': xgb,
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
#     reg = Reg_HpoSearch(data, target[:, 0], k)
#     best, trials = trail_run(reg.objective, chs.hyperopt_parameters, max_evals=5)
#     print(best)


METRIC_DICT = {
    'mae': mean_absolute_error,
    'r2': r2_score,
    'mse': mean_squared_error,
}


class Reg_HpoSearch(object):
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
        loss = score
        return {'loss': loss, 'status': STATUS_OK}

    def reg_objective(self, args):
        print("Training with params: ")
        print(args)
        reg = REG_DICT[self.model_name](**args)
        reg.fit(self.x_train, self.y_train)
        print(self.x_train.shape, self.y_train.shape)
        # validationデータを使用して、ラベルの予測
        pred = reg.predict(self.x_test)
        # 予測ラベルと正解ラベルを使用してmicro f1を計算
        score = self.metric(self.y_test, pred)
        # 今回はmicro f1を最大化したいので、-1をかけて最小化に合わせる
        print("\t{0} {1}\n\n".format(self.metric.__qualname__, score))
        return {'loss': 1 * score, 'status': STATUS_OK}
