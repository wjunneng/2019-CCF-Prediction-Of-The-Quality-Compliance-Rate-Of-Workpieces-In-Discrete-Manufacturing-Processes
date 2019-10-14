import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek


class Sampling(object):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, ratio: dict):
        self.X = X
        self.y = y
        self.ratio = ratio

    def smote(self, **params) -> (pd.DataFrame, pd.DataFrame):
        """
        过采样
        :param params:
        :return:
        """
        # before
        print('Before resampled dataset shape %s' % Counter(self.y))

        X_columns = list(self.X.columns)

        # smote 算法
        smote = SMOTE(ratio=self.ratio, n_jobs=10, kind='svm')
        self.X, self.y = smote.fit_sample(self.X, self.y)

        print('Before resampled dataset shape %s' % Counter(self.y))

        # 重新构造DataFrame
        self.X = pd.DataFrame(data=self.X, columns=X_columns)
        self.y = pd.Series(data=self.y)

        return self.X, self.y

    def smote_enn(self, **params) -> (pd.DataFrame, pd.DataFrame):
        """
        过采样+欠采样
        :return:
        """
        # before
        print('Before resampled dataset shape %s' % Counter(self.y))

        X_columns = list(self.X.columns)

        # smoteenn 算法
        smote = SMOTEENN(ratio=self.ratio, n_jobs=10)
        self.X, self.y = smote.fit_sample(self.X, self.y)

        print('Before resampled dataset shape %s' % Counter(self.y))
        # 重新构造DataFrame
        self.X = pd.DataFrame(data=self.X, columns=X_columns)
        self.y = pd.Series(data=self.y)

        return self.X, self.y

    def smote_tomek(self, params) -> (pd.DataFrame, pd.DataFrame):
        """
        过采样+欠采样
        :return:
        """
        # before
        print('Before resampled dataset shape %s' % Counter(self.y))

        X_columns = list(self.X.columns)

        # smotetomek 算法
        smote = SMOTETomek(ratio=self.ratio, n_jobs=10)
        self.X, self.y = smote.fit_sample(self.X, self.y)
        print('Before resampled dataset shape %s' % Counter(self.y))

        # 重新构造DataFrame
        self.X = pd.DataFrame(data=self.X, columns=X_columns)
        self.y = pd.Series(data=self.y)

        return self.X, self.y
