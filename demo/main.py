from util import *
from config import *


def main():
    import time

    start = time.clock()

    # 获取训练集，标签列，测试集，group
    X_train, y_train, X_test, testing_group = preprocessing()
    print('\n数据预处理 耗时： %s \n' % str(time.clock() - start))

    for column in DefaultConfig.encoder_columns:
        print(X_test[column + '_bin'].value_counts())
        print(X_train[column + '_bin'].value_counts())

    if DefaultConfig.select_model is 'lgb':
        lgb_model(X_train, y_train, X_test, testing_group)
    elif DefaultConfig.select_model is 'ctb':
        ctb_model(X_train, y_train, X_test, testing_group)

    print('\n模型训练与预测 耗时： %s \n' % str(time.clock() - start))
    print(time.clock() - start)


if __name__ == '__main__':
    main()
