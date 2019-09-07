# -*- coding: utf-8 -*-
"""
    配置文件
"""
import os


class DefaultConfig(object):
    """
    参数配置
    """

    def __init__(self):
        pass

    # 项目路径
    project_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])

    # first_round_testing_data
    first_round_testing_data_path = project_path + '/data/original/first_round_testing_data.csv'
    # first_round_training_data
    first_round_training_data_path = project_path + '/data/original/first_round_training_data.csv'
    # submit_example
    submit_example_path = project_path + '/data/original/submit_example.csv'

    # 原始特征列
    original_columns = ['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4', 'Parameter5', 'Parameter6',
                        'Parameter7', 'Parameter8', 'Parameter9', 'Parameter10']
    # 处理异常类别变量
    outlier_columns = ['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4']
    # label_encoder类别变量
    encoder_columns = ['Parameter5', 'Parameter6', 'Parameter7', 'Parameter8', 'Parameter9', 'Parameter10']
    # label列
    label_columns = ['Parameter5_label', 'Parameter6_label', 'Parameter7_label', 'Parameter8_label', 'Parameter9_label',
                     'Parameter10_label']

    # select_model
    select_model = 'lgb'
    # select_model = 'cbt'

    # 一、
    # add_numerical_feature_no_replace
    no_replace_add_numerical_feature = True
    # add_numerical_feature_df_cache
    df_add_numerical_feature_cache_path = project_path + '/data/cache/df_add_numerical_feature.h5'

    # add_label_feature_no_replace
    no_replace_add_label_feature = False
    # add_label_feature_df_cache
    df_add_label_feature_cache_path = project_path + '/data/cache/df_add_label_feature.h5'

    # 二、
    # convert_no_replace
    no_replace_convert = False
    # convert_cache
    df_convert_cache_path = project_path + '/data/cache/df_convert.h5'

    # 三、
    # smote_no_replace
    no_replace_smote = False
    # smote_cache
    X_train_smote_cache_path = project_path + '/data/cache/X_train_smote.h5'
    y_train_smote_cache_path = project_path + '/data/cache/y_train_smote.h5'

    single_model = False

    # lgb_submit
    lgb_submit_path = project_path + '/data/submit/lgb_submit.csv'
    # cbt_submit
    cbt_submit_path = project_path + '/data/submit/cbt_submit.csv'

    submit_path = project_path + '/data/submit/submit.csv'

    lgb_feature_cache_path = project_path + '/data/cache/lgb_feature.h5'
