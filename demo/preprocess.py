import pandas as pd
import os

import numpy as np
from configuration.config import DefaultConfig
from utils.utils import Utils
from collections import Counter
import sys


class Preprocess(object):
    """
    预处理类
    """

    def __init__(self, first_round_testing_data_path, first_round_training_data_path):
        self.first_round_testing_data_path = first_round_testing_data_path
        self.first_round_training_data_path = first_round_training_data_path

    def get_first_round_tesing_data(self, **params):
        """
        返回测试集数据
        :param params:
        :return:
        """
        first_round_testing_data = pd.read_csv(DefaultConfig.first_round_testing_data_path)

        return first_round_testing_data

    def get_first_round_training_data(self, **params):
        """
        返回训练集
        :param params:
        :return:
        """

        first_round_training_data = pd.read_csv(self.first_round_training_data_path)

        return first_round_training_data

    def main(self):
        """

        :return:
        """
        # testing data
        first_round_testing_data = self.get_first_round_tesing_data()
        # training data
        first_round_training_data = self.get_first_round_training_data()

        # Quality_label
        first_round_training_data['Quality_label'] = first_round_training_data['Quality_label'].apply(
            lambda x: 3 if x == 'Fail' else x)
        first_round_training_data['Quality_label'] = first_round_training_data['Quality_label'].apply(
            lambda x: 2 if x == 'Pass' else x)
        first_round_training_data['Quality_label'] = first_round_training_data['Quality_label'].apply(
            lambda x: 1 if x == 'Good' else x)
        first_round_training_data['Quality_label'] = first_round_training_data['Quality_label'].apply(
            lambda x: 0 if x == 'Excellent' else x)

        # group特征列
        testing_group = first_round_testing_data['Group']

        # 测试集
        X_test = first_round_testing_data.loc[:, DefaultConfig.parameter_features]

        # 训练集
        X_train = first_round_training_data.loc[:, DefaultConfig.parameter_features + DefaultConfig.attribute_features]
        # 标签列
        y_train = first_round_training_data['Quality_label']

        # 一、
        # 数据进行规范化
        df = normalization(X_train, X_test)

        # 二、
        if DefaultConfig.select_model is not 'cbt':
            # 添加数值列
            df = add_numerical_feature(df, X_train, y_train)
        # 处理类别变量
        df = add_label_feature(df, X_train, y_train, X_test)

        # 三、
        # 重新获取X_train
        X_train = df[:X_train.shape[0]]
        print('X_train.shape: ', X_train.shape)
        # 重新获取X_test
        X_test = df[X_train.shape[0]:X_train.shape[0] + y_train.shape[0]]
        print('X_test.shape: ', X_test.shape)

        print('X_train.na:')
        print(np.where(pd.isna(X_train)))
        print('X_test.na:')
        print(np.where(pd.isna(X_test)))

        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        # 三、
        if DefaultConfig.select_model is 'lgb':
            # 过采样+欠采样 效果好
            X_train, y_train = smote(X_train=X_train, y_train=y_train)

            # 过采样后整型数据会变成浮点型数据
            for column in X_train.columns:
                if '_label' in column:
                    X_train[column] = X_train[column].astype(int)
                    X_test[column] = X_test[column].astype(int)

        X_test.to_hdf(path_or_buf=DefaultConfig.X_test_cache_path, mode='w', key='X_test')

        return X_train, y_train, X_test, testing_group
