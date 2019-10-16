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

    # #########################################
    # wyt
    # select_model = 'lgbm'
    # zya
    select_model = 'cbt'
    # wjunneng
    # select_model = 'merge'

    # #########################################
    no_replace = False
    X_train_cache_path = project_path + '/data/cache/X_train_cache.h5'
    y_train_cache_path = project_path + '/data/cache/y_train_cache.h5'
    X_test_cache_path = project_path + '/data/cache/X_test_cache.h5'

    # lgb_submit
    lgb_submit_path = project_path + '/data/submit/lgb_submit.csv'
    # cbt_submit
    cbt_submit_path = project_path + '/data/submit/cbt_submit.csv'

    submit_path = project_path + '/data/submit/submit.csv'

    lgb_feature_cache_path = project_path + '/data/cache/lgb_feature.h5'
    cbt_feature_cache_path = project_path + '/data/cache/cbt_feature.h5'

    # ########################################## single
    single_model = True

