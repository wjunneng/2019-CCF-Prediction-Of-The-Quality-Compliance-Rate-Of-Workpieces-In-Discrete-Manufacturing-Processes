from util import *
from config import *

def main():
    import time

    start = time.clock()

    # 获取训练集，标签列，测试集，group
    X_train, y_train, X_test, testing_group = preprocessing()
    print('\n数据预处理 耗时： %s \n' % str(time.clock() - start))

    # svm_model(X_train, y_train, X_test, testing_group)
    lgb_model(X_train, y_train, X_test, testing_group)
    print('\n模型构建 耗时： %s \n' % str(time.clock() - start))


if __name__ == '__main__':
    main()
