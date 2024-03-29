from demo.preprocess import Preprocess
from configuration.config import *
from model.cbt import CatBoost
from model.lgbm import LightGbm

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings('ignore')


def main():
    import time

    start = time.clock()

    # 获取训练集，标签列，测试集，group
    X_train, y_train, X_test, testing_group = Preprocess(
        first_round_training_data_path=DefaultConfig.first_round_training_data_path,
        first_round_testing_data_path=DefaultConfig.first_round_testing_data_path).main()
    print('\n数据预处理 耗时： %s \n' % str(time.clock() - start))

    columns = X_train.columns
    print(list(columns))
    # ############################################# cbt
    if DefaultConfig.select_model is 'cbt':
        columns = ['Parameter10', 'Parameter9', 'Parameter8', 'Parameter5', 'Parameter6', 'Parameter7',
                   'Parameter5_count_encoded', 'Parameter6_count_encoded', 'Parameter7_count_encoded',
                   'Parameter8_count_encoded', 'Parameter9_count_encoded', 'Parameter10_count_encoded']
        X_train = X_train[columns]
        X_test = X_test[columns]

    # ############################################# lgb
    if DefaultConfig.select_model is 'lgbm':
        columns = [
            'Parameter4_groupby_Parameter10_mean_ratio',
            'Parameter4',
            'Parameter10',
            'Parameter1',
            'Parameter2',
            'Parameter5',
            'inv(div(max(min(X3, X1), abs(min(X3, X1))), log(div(log(max(min(X3, X1), abs(log(div(max(min(X3, X1), abs(log(sqrt(X2)))), log(div(log(max(min(X3, X1), abs(log(max(X0, X2))))), log(mul(-0.088, X2))))))))), log(mul(-0.088, X2))))))',
            'inv(div(max(min(X3, X1), abs(log(inv(X2)))), log(div(log(max(min(X3, X1), abs(log(max(X0, X2))))), log(mul(-0.088, X2))))))',
            'inv(div(max(sqrt(X2), abs(log(inv(max(X0, X2))))), log(div(log(max(min(X3, X1), abs(log(max(X0, X2))))), log(div(sqrt(min(X3, X1)), log(mul(-0.088, X2))))))))',
            'div(sqrt(sub(X2, X3)), div(max(min(X3, X1), inv(inv(add(inv(sqrt(X2)), X2)))), log(div(sqrt(min(X3, X1)), log(mul(-0.088, X2))))))',
            'Parameter3',
            'Parameter6',
            'Parameter9',
            'Parameter8',
            'Parameter7',
            'Parameter10_label'
        ]

        X_train = X_train[columns]
        X_test = X_test[columns]

    print('select_columns: ', list(columns))
    if DefaultConfig.select_model is 'lgb':
        LightGbm(X_train, y_train, X_test).main(testing_group)
    elif DefaultConfig.select_model is 'cbt':
        CatBoost(X_train, y_train, X_test).main(testing_group)

    print('\n模型训练与预测 耗时： %s \n' % str(time.clock() - start))
    print(time.clock() - start)

    if DefaultConfig.select_model is 'merge':
        merge()


if __name__ == '__main__':
    main()
