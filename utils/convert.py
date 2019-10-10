from sklearn.preprocessing import PowerTransformer, FunctionTransformer
import pandas as pd
import numpy as np


class Convert(object):
    def __init__(self, df: pd.DataFrame, columns: list):
        self.df = df
        self.columns = columns

    def yeo_johnson(self, **params) -> pd.DataFrame:
        """
        yeo_johnson变换
        :return:
        """
        print('进行yeo-johnson变换的特征列：')
        print(self.columns)

        pt = PowerTransformer(method='yeo-johnson', standardize=True)

        self.df[self.columns] = pt.fit_transform(self.df[self.columns].values)

        return self.df

    def log1p(self, **params) -> pd.DataFrame:
        """
        log1p 变换
        :param params:
        :return:
        """
        print('进行对数变换的特征列：')
        print(self.columns)

        transformer = FunctionTransformer(np.log1p)
        self.df[self.columns] = transformer.fit_transform(self.df[self.columns].values)

        return self.df
