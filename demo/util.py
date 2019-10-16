from utils.utils import Utils
from utils.mean_encoder import MeanEncoder
from configuration.config import DefaultConfig

import pandas as pd
import os


def deal_outlier(X_train, X_test, **params):
    """
    处理异常值
    :param X_train:
    :param X_test:
    :param params:
    :return:
    """
    params = ['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4', 'Parameter5', 'Parameter6', 'Parameter7',
              'Parameter8', 'Parameter9', 'Parameter10', 'Attribute1', 'Attribute2', 'Attribute3', 'Attribute4',
              'Attribute5', 'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'Attribute10']

    max_limit = [3.9e+09, 1.4e+09, 2.9e+09, 3.7e+08, 70, 43, 2.4e+04, 7.6e+04, 6.1e+08, 1.5e+04,
                 1.2e+07, 3.2e+08, 5.1e+09, 6.3e+07, 6.4e+09, 2.6e+07, 8.5e+09, 5.6e+10, 1.8e+12, 2.0e+11]

    max_limit_params = dict(zip(params, max_limit))
    for column in params:
        tmp = max_limit_params[column]
        if column in list(X_train.columns):
            X_train[column] = X_train[column].apply(lambda x: tmp if x > tmp else x)
        if column in list(X_test.columns):
            X_test[column] = X_test[column].apply(lambda x: tmp if x > tmp else x)

    df = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    return df


def add_label_feature(df, X_train, y_train, X_test, **params):
    """
    添加类别特征
    :param df:
    :param params:
    :return:
    """
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

    return df


def count_encode(X, categorical_features, normalize=False):
    """
    计数编码
    :param X:
    :param categorical_features:
    :param normalize:
    :return:
    """
    import numpy as np

    print('Count encoding: {}'.format(categorical_features))
    X_ = pd.DataFrame()
    for cat_feature in categorical_features:
        X_[cat_feature] = X[cat_feature].astype('object').map(X[cat_feature].value_counts())
        if normalize:
            X_[cat_feature] = X_[cat_feature] / np.max(X_[cat_feature])
    X_ = X_.add_suffix('_count_encoded')
    if normalize:
        X_ = X_.astype(np.float32)
        X_ = X_.add_suffix('_normalized')
    else:
        X_ = X_.astype(np.uint32)

    df = pd.concat([X, X_], ignore_index=True, axis=1)
    df.columns = list(X.columns) + list(X_.columns)
    return df


def save_result(testing_group, prediction, **params):
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


def merge(lgb_rate, cbt_rate, **params):
    """
    merge
    :param params:
    :return:
    """
    lgb_submit = pd.read_csv(filepath_or_buffer=DefaultConfig.lgb_submit_path)
    cbt_submit = pd.read_csv(filepath_or_buffer=DefaultConfig.cbt_submit_path)

    # lgb_submit['sum'] = 0
    for column in DefaultConfig.columns:
        lgb_submit[column] = (lgb_rate * lgb_submit[column] + cbt_rate * cbt_submit[column])

    lgb_submit.to_csv(DefaultConfig.submit_path, encoding='utf-8', index=None)
