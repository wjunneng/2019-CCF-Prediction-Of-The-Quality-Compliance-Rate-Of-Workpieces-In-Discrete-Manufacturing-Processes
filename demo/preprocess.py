from configuration.config import DefaultConfig
from utils.mean_encoder import MeanEncoder
from utils.utils import Utils
from utils.sampling import Sampling
from utils.convert import Convert
from utils.numerical_feature import NumericalFeature

from demo.util import deal_outlier
from demo.util import add_label_feature
from demo.util import count_encode

import os
import numpy as np
import pandas as pd


class Preprocess(object):
    """
    预处理类
    """

    def __init__(self, first_round_testing_data_path, first_round_training_data_path):
        self.first_round_testing_data_path = first_round_testing_data_path
        self.first_round_training_data_path = first_round_training_data_path

        # testing data
        self.first_round_testing_data = self.get_first_round_tesing_data()
        # training data
        self.first_round_training_data = self.get_first_round_training_data()

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

        # Quality_label
        first_round_training_data['Quality_label'] = first_round_training_data['Quality_label'].apply(
            lambda x: 0 if x == 'Excellent' else x)
        first_round_training_data['Quality_label'] = first_round_training_data['Quality_label'].apply(
            lambda x: 1 if x == 'Good' else x)
        first_round_training_data['Quality_label'] = first_round_training_data['Quality_label'].apply(
            lambda x: 2 if x == 'Pass' else x)
        first_round_training_data['Quality_label'] = first_round_training_data['Quality_label'].apply(
            lambda x: 3 if x == 'Fail' else x)

        return first_round_training_data

    def pre_deal(self, X_train, y_train, X_test, save=True):
        """
        预先处理
        :return:
        """
        """
        数据进行规范化
        :param df:
        :param params:
        :return:
        """
        X_train_cache_path = DefaultConfig.X_train_cache_path
        y_train_cache_path = DefaultConfig.y_train_cache_path
        X_test_cache_path = DefaultConfig.X_test_cache_path

        if os.path.exists(X_train_cache_path) and os.path.exists(y_train_cache_path) and os.path.exists(
                X_test_cache_path) and DefaultConfig.no_replace:
            X_train = Utils.reduce_mem_usage(pd.read_hdf(path_or_buf=X_train_cache_path, key='X_train', mode='r'))
            # y_train = Utils.reduce_mem_usage(pd.read_hdf(path_or_buf=y_train_cache_path, key='y_train', mode='r'))
            X_test = Utils.reduce_mem_usage(pd.read_hdf(path_or_buf=X_test_cache_path, key='X_test', mode='r'))
        else:
            df = None
            steps = None
            if DefaultConfig.select_model is 'lgbm':
                steps = ['step_1', 'step_2', 'step_3', 'step_4', 'step_5']

            elif DefaultConfig.select_model is 'cbt':
                steps = ['step_1', 'step_2', 'step_5', 'step_6']

            if 'step_1' in steps:
                # ########################################### 要进行范围限制的特征列
                df = deal_outlier(X_train=X_train, X_test=X_test)

            if 'step_2' in steps:
                # ########################################### 要进行yeo-johnson变换的特征列
                columns = DefaultConfig.parameter_numerical_features
                df = Convert(df=df, columns=columns).yeo_johnson()

            if 'step_3' in steps:
                # ########################################### 添加数值列
                df = NumericalFeature(df=df, X_train=X_train, y_train=y_train). \
                    symbolic_transformer(generations=20, n_components=4,
                                         columns=DefaultConfig.parameter_numerical_features)

            if 'step_4' in steps:
                # ########################################### 添加类别列
                df = add_label_feature(df=df, X_train=X_train, y_train=y_train, X_test=X_test)

            if 'step_5' in steps:
                # ########################################### 添加类别列
                df = count_encode(X=df, categorical_features=DefaultConfig.parameter_label_features)

            if 'step_6' in steps:
                # ########################################### 修复
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

            if 'step_7' in steps:
                # ########################################### 样本抽样
                # ['Excellent ratio',   'Good ratio',        'Pass ratio',       'Fail ratio']
                # 0.1711159138196257    0.2560259533930791  0.4284821074119886  0.14437602537530658
                ratio = {0: 1434, 1: 1857, 2: 3387, 3: 1020}

                # 过采样+欠采样 效果好
                X_train, y_train = Sampling(X=X_train, y=y_train, ratio=ratio).smote()

                # 过采样后整型数据会变成浮点型数据
                for column in X_train.columns:
                    if '_label' in column:
                        X_train[column] = X_train[column].astype(int)
                        X_test[column] = X_test[column].astype(int)
            y_train = pd.DataFrame(y_train)
            # if save:
            #     X_train.to_hdf(path_or_buf=X_train_cache_path, mode='w', key='X_train')
            #     y_train.to_hdf(path_or_buf=y_train_cache_path, mode='w', key='y_train')
            #     X_test.to_hdf(path_or_buf=X_test_cache_path, mode='w', key='X_test')

        return X_train, y_train, X_test

    def main(self):
        """
        主函数
        :return:
        """
        # group特征列
        testing_group = self.first_round_testing_data['Group']

        # 测试集
        X_test = self.first_round_testing_data.loc[:, DefaultConfig.parameter_features]

        # 训练集
        X_train = self.first_round_training_data.loc[:,
                  DefaultConfig.parameter_features + DefaultConfig.attribute_features]
        # 标签列
        y_train = self.first_round_training_data['Quality_label']

        X_train, y_train, X_test = self.pre_deal(X_train=X_train, y_train=y_train, X_test=X_test)

        return X_train, y_train, X_test, testing_group
