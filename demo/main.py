from util import *
from config import *
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings('ignore')


def main():
    import time

    start = time.clock()

    # 获取训练集，标签列，测试集，group
    X_train, y_train, X_test, testing_group = preprocess()
    print('\n数据预处理 耗时： %s \n' % str(time.clock() - start))

    for column in ['Parameter7_pred_3', 'Parameter8_pred_0', 'Parameter9_label', 'Parameter8_pred_2',
                   'Parameter8_pred_3', 'Parameter8_pred_1']:
        if column in X_test.columns and column in X_train.columns:
            del X_train[column]
            del X_test[column]

    print(list(X_train.columns))
    print(len(X_train.columns))
    if DefaultConfig.select_model is 'lgb':
        lgb_model(X_train, y_train, X_test, testing_group)
    elif DefaultConfig.select_model is 'cbt':
        cbt_model(X_train, y_train, X_test, testing_group)

    print('\n模型训练与预测 耗时： %s \n' % str(time.clock() - start))
    print(time.clock() - start)

    merge()


if __name__ == '__main__':
    main()
