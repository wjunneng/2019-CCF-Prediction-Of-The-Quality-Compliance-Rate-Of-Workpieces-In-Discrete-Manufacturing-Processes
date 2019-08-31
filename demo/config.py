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

    # 处理异常类别变量
    outlier_columns = ['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4']
    # label_encoder类别变量
    encoder_columns = ['Parameter5', 'Parameter6', 'Parameter7', 'Parameter8', 'Parameter9', 'Parameter10']
    # label_columns
    label_columns = ['Parameter5_label', 'Parameter6_label', 'Parameter7_label', 'Parameter8_label', 'Parameter9_label']

    # no_replace
    no_replace = False

    # select_model
    select_model = 'lgb'
    # select_model = 'ctb'
