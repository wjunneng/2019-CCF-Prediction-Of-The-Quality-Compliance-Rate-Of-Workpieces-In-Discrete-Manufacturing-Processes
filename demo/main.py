from util import *


def main():
    from collections import Counter
    import time
    import pandas as pd

    start = time.clock()

    # 获取训练集，标签列，测试集，group
    X_train, y_train, X_test, testing_group = preprocessing()

    import numpy as np
    print(np.where(np.isnan(X_train)))
    print(np.where(np.isnan(X_test)))

    # 过采样+欠采样
    from imblearn.combine import SMOTETomek
    smote_tomek = SMOTETomek(ratio={0: 3000, 1: 3000, 2: 3000, 3: 3000}, random_state=0, n_jobs=10)
    train_X, train_y = smote_tomek.fit_sample(X_train, y_train)

    print('Resampled dataset shape %s' % Counter(train_y))

    # X_train
    X_train = pd.DataFrame(data=train_X, columns=X_train.columns)

    # svm_model(X_train, pd.Series(train_y), X_test, testing_group)
    lgb_model(X_train, pd.Series(train_y), X_test, testing_group)

    print(time.clock() - start)


if __name__ == '__main__':
    main()
