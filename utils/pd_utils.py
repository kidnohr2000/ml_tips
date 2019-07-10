# -*- coding: utf-8 -*-
# vim:tabstop=4:shiftwidth=4:expandtab

# 複数のモデルの計算結果の傾向より、分類を予測するメタモデル

import pandas as pd
import numpy as np
# from IPython.core.debugger import Pdb; Pdb().set_trace()


def isinstance_pd(X):
    return isinstance(X, (pd.DataFrame, pd.Series))


def get_column_specific(df, name, _type= 'start', cols=[]):
    if _type == 'end':
        return df[cols + [l for l in df.columns if l.endswith(name)]]
    else:
        return df[cols + [l for l in df.columns if l.startswith(name)]]