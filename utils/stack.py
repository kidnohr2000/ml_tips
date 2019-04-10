# -*- coding: utf-8 -*-
# vim:tabstop=4:shiftwidth=4:expandtab

# 複数のモデルの計算結果の傾向より、分類を予測するメタモデル

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone


class BinaryClassifiyStackingModels(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, meta_prams={}, n_folds=5):
        self.base_models = base_models
        self.meta_base_model = meta_model
        self.meta_model = meta_model(**meta_prams)
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=123)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kf.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict_proba(X[holdout_index])[:, 1]
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict_proba(X)[:, 1] for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

    def best_params_(self, X, y, param_grid):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=123)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kf.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict_proba(X[holdout_index])[:, 1]
                out_of_fold_predictions[holdout_index, i] = y_pred

        _model = GridSearchCV(self.meta_base_model(), param_grid, cv=kf.split(out_of_fold_predictions, y), scoring='f1', iid=True)

        _model.fit(x_train, y_train)
        return _model.best_params_


class MultiClassifiyStackingModels(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, meta_params={}, n_folds=5, n_targets=5):
        self.base_models = base_models
        self.meta_base_model = meta_model
        self.meta_model = meta_model(**meta_params)
        self.n_folds = n_folds
        self.n_targets = n_targets

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=123)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models) * self.n_targets))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kf.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict_proba(X[holdout_index])
                out_of_fold_predictions[holdout_index,  i * self.n_targets: (i + 1) * self.n_targets] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.array([model.predict_proba(X) for model in base_models]).mean(axis=0)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

    def best_params_(self, X, y, param_grid):
        self.base_models_ = [list() for x in self.base_models]
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=123)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models) * self.n_targets))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kf.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict_proba(X[holdout_index])
                out_of_fold_predictions[holdout_index,  i * self.n_targets: (i + 1) * self.n_targets] = y_pred

        _model = GridSearchCV(self.meta_base_model(), param_grid, cv=kf.split(out_of_fold_predictions, y), scoring='f1_micro', iid=True)

        _model.fit(x_train, y_train)
        return _model.best_params_
