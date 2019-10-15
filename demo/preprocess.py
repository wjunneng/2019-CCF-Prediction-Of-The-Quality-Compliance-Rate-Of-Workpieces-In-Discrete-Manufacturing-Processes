from configuration.config import DefaultConfig
from utils.mean_encoder import MeanEncoder
from utils.utils import Utils
from utils.sampling import Sampling
from utils.convert import Convert

import os
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer


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

    def normalization(self, X_train, X_test, save=True, **params):
        """
        数据进行规范化
        :param df:
        :param params:
        :return:
        """
        path = DefaultConfig.df_normalization_cache_path

        if os.path.exists(path) and DefaultConfig.no_replace_normalization:
            df = Utils.reduce_mem_usage(pd.read_hdf(path_or_buf=path, key='normalization', mode='r'))
        else:
            # ########################################### 要进行范围限制的特征列
            max_limit = [3.9e+09, 1.4e+09, 2.9e+09, 3.7e+08, 70, 43, 2.4e+04, 7.6e+04, 6.1e+08, 1.5e+04,
                         1.2e+07, 3.2e+08, 5.1e+09, 6.3e+07, 6.4e+09, 2.6e+07, 8.5e+09, 5.6e+10, 1.8e+12, 2.0e+11]

            params = ['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4', 'Parameter5', 'Parameter6', 'Parameter7',
                      'Parameter8', 'Parameter9', 'Parameter10', 'Attribute1', 'Attribute2', 'Attribute3', 'Attribute4'
                , 'Attribute5', 'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'Attribute10']

            max_limit_params = dict(zip(params, max_limit))

            # 处理异常值
            for column in params:
                tmp = max_limit_params[column]
                X_train[column] = X_train[column].apply(lambda x: tmp if x > tmp else x)
                if column in list(X_test.columns):
                    X_test[column] = X_test[column].apply(lambda x: tmp if x > tmp else x)

            df = pd.concat([X_train, X_test], axis=0, ignore_index=True)

            # ########################################### 要进行yeo-johnson变换的特征列
            print('进行yeo-johnson变换的特征列：')
            print(DefaultConfig.parameter_numerical_features)

            df[DefaultConfig.parameter_numerical_features] = Convert(df=df, columns=DefaultConfig.
                                                                     parameter_numerical_features).yeo_johnson()
            if save:
                df.to_hdf(path_or_buf=path, key='normalization')

        return df

    def add_numerical_feature(self, df, X_train, y_train, save=True, **params):
        """
        添加数值特征
        :param df:
        :param X_train:
        :param y_train:
        :param X_test:
        :param save:
        :param params:
        :return:
        """
        path = DefaultConfig.df_add_numerical_feature_cache_path

        if os.path.exists(path) and DefaultConfig.no_replace_add_numerical_feature:
            df = Utils.reduce_mem_usage(pd.read_hdf(path_or_buf=path, key='add_numerical_feature', mode='r'))
        else:
            # ###########################################  添加数值列
            # 生成的特征数
            n_components = 4
            generations = 20

            function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min']

            gp = SymbolicTransformer(generations=generations, population_size=2000,
                                     hall_of_fame=100, n_components=n_components,
                                     function_set=function_set,
                                     parsimony_coefficient=0.0005,
                                     max_samples=0.8, verbose=1,
                                     random_state=0, metric='spearman', n_jobs=10)

            gp.fit(X=X_train[DefaultConfig.parameter_numerical_features], y=y_train)
            gp_features = gp.transform(df[DefaultConfig.parameter_numerical_features])

            columns = list(df.columns)
            for i in range(n_components):
                columns.append(str(gp._best_programs[i]))

            df = pd.DataFrame(data=np.hstack((df.values, gp_features)), columns=columns, index=None)

            if save:
                df.to_hdf(path_or_buf=path, key='add_numerical_feature')

        return df

    def add_label_feature(self, df, X_train, y_train, X_test, save=True, **params):
        """
        添加类别特征
        :param df:
        :param params:
        :return:
        """
        path = DefaultConfig.df_add_label_feature_cache_path

        if os.path.exists(path) and DefaultConfig.no_replace_add_label_feature:
            df = Utils.reduce_mem_usage(pd.read_hdf(path_or_buf=path, key='add_label_feature', mode='r'))
        else:
            # ###########################################  添加新的类别列
            columns = ['Parameter10']
            # 1.均值编码
            for column in columns:
                # 声明需要平均数编码的特征
                MeanEnocodeFeature = [column]
                # 声明平均数编码的类
                ME = MeanEncoder(MeanEnocodeFeature)
                # 对训练数据集的X和y进行拟合
                X_train = ME.fit_transform(X_train, y_train)
                # 对测试集进行编码
                X_test = ME.transform(X_test)

            columns = [i for i in list(X_train.columns) if i not in list(df.columns)]
            df[columns] = pd.concat([X_train[columns], X_test[columns]], axis=0, ignore_index=True)

            # 2.简单地取整数
            for column in ['Parameter5', 'Parameter6', 'Parameter7', 'Parameter8', 'Parameter9', 'Parameter10']:
                df[column + '_label'] = df[column].apply(lambda x: int(round(x)))

            # 3.数值列
            for c_col in ['Parameter10']:
                for n_col in ['Parameter1', 'Parameter4']:
                    df[n_col + '_groupby_' + c_col + '_mean_ratio'] = df[n_col] / (
                        df[c_col].map(df[n_col].groupby(df[c_col]).mean()))

            # ###########################################  添加类别列
            # # 类别列
            for column_i in ['Parameter10']:
                # 数值列
                for column_j in ['Parameter1']:
                    stats = df.groupby(column_i)[column_j].agg(['mean', 'max', 'min', 'std', 'sum'])
                    stats.columns = ['mean_' + column_j, 'max_' + column_j, 'min_' + column_j, 'std_' + column_j,
                                     'sum_' + column_j]
                    df = df.merge(stats, left_on=column_i, right_index=True, how='left')

            # ###########################################  删除属性列
            for column in DefaultConfig.attribute_features:
                del df[column]

            if save:
                df.to_hdf(path_or_buf=path, key='add_label_feature')
        return df

    def save_result(self, testing_group, prediction, **params):
        """
        保存结果
        :param testing_group:
        :param prediction:
        :param params:
        :return:
        """
        df = pd.concat([testing_group, pd.Series(prediction)], axis=1, ignore_index=True)
        df.columns = ['Group', 'predictions']

        result = pd.DataFrame(columns=['Group', 'Excellent ratio', 'Good ratio', 'Pass ratio', 'Fail ratio'])
        for (group, value) in df.groupby(['Group']):
            fail_ratio = value[value['predictions'] == 0].shape[0] / value.shape[0]
            pass_ratio = value[value['predictions'] == 1].shape[0] / value.shape[0]
            good_ratio = value[value['predictions'] == 2].shape[0] / value.shape[0]
            excellent_ratio = value[value['predictions'] == 3].shape[0] / value.shape[0]
            result.loc[result.shape[0]] = [int(group), excellent_ratio, good_ratio, pass_ratio, fail_ratio]

        result['Group'] = result['Group'].astype(int)
        result.to_csv(
            path_or_buf=DefaultConfig.project_path + '/data/submit/' + DefaultConfig.select_model + '_submit.csv',
            index=False, encoding='utf-8')

    def merge(**params):
        """
        merge
        :param params:
        :return:
        """
        lgb_submit = pd.read_csv(filepath_or_buffer=DefaultConfig.lgb_submit_path)
        cbt_submit = pd.read_csv(filepath_or_buffer=DefaultConfig.cbt_submit_path)

        # lgb_submit['sum'] = 0
        for column in DefaultConfig.columns:
            lgb_submit[column] = (0.4 * lgb_submit[column] + 0.6 * cbt_submit[column])
        #     lgb_submit['sum'] += lgb_submit[column]
        #
        # for column in ['Excellent ratio', 'Good ratio', 'Pass ratio', 'Fail ratio']:
        #     lgb_submit[column] /= lgb_submit['sum']
        #
        # del lgb_submit['sum']

        lgb_submit.to_csv(DefaultConfig.submit_path, encoding='utf-8', index=None)

    def caculate_rate(**params):
        """
        计算占比
        :param params:
        :return:
        """
        path = DefaultConfig.submit_path

        df = pd.read_csv(filepath_or_buffer=path, encoding='utf-8')

        # 0.855
        # ['Excellent ratio',   'Good ratio',        'Pass ratio',       'Fail ratio']
        # 0.1711159138196257    0.2560259533930791  0.4284821074119886  0.14437602537530658
        for column in DefaultConfig.columns:
            print(df[column].mean())

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
        df = self.normalization(X_train, X_test)

        # 二、
        if DefaultConfig.select_model is not 'cbt':
            # 添加数值列
            df = self.add_numerical_feature(df, X_train, y_train)
        # 处理类别变量
        df = self.add_label_feature(df, X_train, y_train, X_test)

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
            ratio = {0: 1434, 1: 1857, 2: 3387, 3: 1020}

            # 过采样+欠采样 效果好
            X_train, y_train = Sampling(X=X_train, y=y_train, ratio=ratio).smote()

            # 过采样后整型数据会变成浮点型数据
            for column in X_train.columns:
                if '_label' in column:
                    X_train[column] = X_train[column].astype(int)
                    X_test[column] = X_test[column].astype(int)

        X_test.to_hdf(path_or_buf=DefaultConfig.X_test_cache_path, mode='w', key='X_test')

        return X_train, y_train, X_test, testing_group
