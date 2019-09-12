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

    # ######################################### Parameter
    # Parameter
    parameter_features = ['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4', 'Parameter5', 'Parameter6',
                          'Parameter7', 'Parameter8', 'Parameter9', 'Parameter10']
    # numerical
    parameter_numerical_features = ['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4']
    # label
    parameter_label_features = ['Parameter5', 'Parameter6', 'Parameter7', 'Parameter8', 'Parameter9', 'Parameter10']

    # columns
    columns = ['Excellent ratio', 'Good ratio', 'Pass ratio', 'Fail ratio']

    # ######################################### Attribute
    # attribute
    attribute_features = ['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5', 'Attribute6',
                          'Attribute7', 'Attribute8', 'Attribute9', 'Attribute10']
    # numerical
    attribute_numerical_features = ['Attribute1', 'Attribute2', 'Attribute3']
    # label
    attribute_label_features = ['Attribute4', 'Attribute5', 'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9',
                                'Attribute10']

    # ######################################### select_model
    # wyt
    # select_model = 'lgb'
    # zya
    # select_model = 'cbt'
    # wjunneng
    select_model = 'merge'

    # ######################################### 一、normalization
    no_replace_normalization = True
    df_normalization_cache_path = project_path + '/data/cache/df_normalization.h5'

    # ######################################### 二、add_numerical_feature/add_label_feature
    # add_numerical_feature_no_replace
    no_replace_add_numerical_feature = True
    # add_numerical_feature_df_cache
    df_add_numerical_feature_cache_path = project_path + '/data/cache/df_add_numerical_feature.h5'
    # add_label_feature_no_replace
    no_replace_add_label_feature = False
    # add_label_feature_df_cache
    df_add_label_feature_cache_path = project_path + '/data/cache/df_add_label_feature.h5'

    # ######################################### 四、smote
    # smote_no_replace
    no_replace_smote = False
    # smote_cache
    X_train_smote_cache_path = project_path + '/data/cache/X_train_smote.h5'
    y_train_smote_cache_path = project_path + '/data/cache/y_train_smote.h5'

    # lgb_submit
    lgb_submit_path = project_path + '/data/submit/lgb_submit.csv'
    # cbt_submit
    cbt_submit_path = project_path + '/data/submit/cbt_submit.csv'

    submit_path = project_path + '/data/submit/submit.csv'

    lgb_feature_cache_path = project_path + '/data/cache/lgb_feature.h5'
    cbt_feature_cache_path = project_path + '/data/cache/cbt_feature.h5'
