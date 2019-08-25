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

    # no_replace
    no_replace = False

    # select_model
    select_model = 'lgb'
