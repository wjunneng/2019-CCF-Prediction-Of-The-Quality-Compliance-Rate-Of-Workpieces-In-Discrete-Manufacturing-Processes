from config import DefaultConfig


def get_first_round_tesing_data(**params):
    """
    返回测试集数据
    :param params:
    :return:
    """
    import pandas as pd

    first_round_testing_data = pd.read_csv(DefaultConfig.first_round_testing_data_path)

    return first_round_testing_data


def get_first_round_training_data(**params):
    """
    返回训练集
    :param params:
    :return:
    """
    import pandas as pd

    first_round_training_data = pd.read_csv(DefaultConfig.first_round_training_data_path)

    return first_round_training_data


def max_min_scalar(df, **params):
    """
    归一化
    :param df:
    :param params:
    :return:
    """
    max_limit = [3.9e+09, 1.4e+09, 2.9e+09, 3.7e+08, 70, 43, 2.4e+04, 7.6e+04, 6.1e+08, 1.5e+04]
    params = ['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4', 'Parameter5', 'Parameter6', 'Parameter7',
              'Parameter8', 'Parameter9', 'Parameter10']

    max_limit_params = dict(zip(params, max_limit))

    for column in df.columns:
        tmp = max_limit_params[column]
        df[column] = df[column].apply(lambda x: tmp if x > tmp else x)

    return df

    ################################################### 归一化 ##########################################################
    # import pandas as pd
    # from sklearn.preprocessing import MaxAbsScaler
    # # 保存columns
    # columns = df.columns
    # df = MaxAbsScaler().fit_transform(df)
    # df = pd.DataFrame(data=df, columns=columns)

    for column in DefaultConfig.outlier_columns:
        df[column] = df[column].apply(lambda x: np.log(x+1))

    return df


def deal_outlier(df, **params):
    """
    处理异常值
    :param df:
    :param params:
    :return:
    """
    Q1 = df.describe().ix['25%', :].sort_index()
    Q3 = df.describe().ix['75%', :].sort_index()

    Q1_dict = Q1.to_dict()
    Q3_dict = Q3.to_dict()

    min = (Q3 - 3 * (Q3 - Q1)).to_dict()
    max = (Q3 + 3 * (Q3 - Q1)).to_dict()

    for column in DefaultConfig.outlier_columns:
        df[column] = df[column].apply(lambda x: Q1_dict[column] if x <= min[column] else x)
        df[column] = df[column].apply(lambda x: Q3_dict[column] if x >= max[column] else x)

    return df


def smote(X_train, y_train, **params):
    """
    过采样+欠采样
    :param X_train:
    :param y_train:
    :param params:
    :return:
    """
    import pandas as pd
    from collections import Counter

    # 过采样+欠采样
    from imblearn.combine import SMOTETomek
    smote_tomek = SMOTETomek(ratio={0: 2000, 1: 2000, 2: 3000, 3: 2000}, random_state=0, n_jobs=10)
    train_X, train_y = smote_tomek.fit_sample(X_train, y_train)

    # from imblearn.over_sampling import SMOTE
    # smote = SMOTE(ratio={0: 2000, 1: 2000, 2: 3000, 3: 2000}, n_jobs=10)
    # train_X, train_y = smote.fit_sample(X_train, y_train)
    print('Resampled dataset shape %s' % Counter(train_y))

    # X_train
    X_train = pd.DataFrame(data=train_X, columns=X_train.columns)

    return X_train, pd.Series(train_y)


def add_feature(df, **params):
    """
    添加新的类别特征
    :param df:
    :param params:
    :return:
    """
    import numpy as np
    from sklearn import preprocessing

    # 添加新的类别列
    for column in DefaultConfig.encoder_columns:
        df[column + '_label'] = df[column].apply(lambda x: str(round(x)))

    # 获取要进行label_encoder的特征列
    object_cols = list(df.dtypes[df.dtypes == np.object].index)

    # 进行label_encoder
    print('one_hot 处理的特征列： %s' % ' '.join(object_cols))
    lbl = preprocessing.LabelEncoder()
    for col in object_cols:
        df[col] = lbl.fit(df[col].astype('str')).transform(df[col].astype('str'))

    return df


def preprocessing(**params):
    """
    数据预处理
    :param params:
    :return:
    """
    import pandas as pd

    # testing data
    first_round_testing_data = get_first_round_tesing_data()
    # training data
    first_round_training_data = get_first_round_training_data()

    # 选中的特征列
    select_columns = ['Parameter1', 'Parameter4', 'Parameter2', 'Parameter3', 'Parameter5', 'Parameter6', 'Parameter7',
                      'Parameter8', 'Parameter9', 'Parameter10']

    # Quality_label
    first_round_training_data['Quality_label'] = first_round_training_data['Quality_label'].apply(
        lambda x: 3 if x == 'Fail' else x)
    first_round_training_data['Quality_label'] = first_round_training_data['Quality_label'].apply(
        lambda x: 2 if x == 'Pass' else x)
    first_round_training_data['Quality_label'] = first_round_training_data['Quality_label'].apply(
        lambda x: 1 if x == 'Good' else x)
    first_round_training_data['Quality_label'] = first_round_training_data['Quality_label'].apply(
        lambda x: 0 if x == 'Excellent' else x)

    # group特征列
    testing_group = first_round_testing_data['Group']

    # 测试集
    X_test = first_round_testing_data[select_columns]

    # 训练集
    X_train = first_round_training_data[select_columns]
    # 标签列
    y_train = first_round_training_data['Quality_label']

    # 待优化，效果很不好
    # 归一化
    X_test = max_min_scalar(X_test)
    X_train = max_min_scalar(X_train)

    # 待优化，效果很不好
    # # 处理异常值
    # result = deal_outlier(pd.concat([X_train, X_test], axis=0, ignore_index=True))
    # # 去除index
    # result.reset_index(inplace=True, drop=True)
    # # 重新获取X_train
    # X_train = result[:X_train.shape[0]]
    # print('X_train.shape: ', X_train.shape)
    # # 重新获取X_test
    # X_test = result[X_train.shape[0]:X_train.shape[0] + y_train.shape[0]]
    # print('X_test.shape: ', X_test.shape)

    # 处理类别变量 提升幅度在0.03左右
    result = add_feature(pd.concat([X_train, X_test], axis=0, ignore_index=True))
    # 去除index
    result.reset_index(inplace=True, drop=True)
    # 重新获取X_train
    X_train = result[:X_train.shape[0]]
    print('X_train.shape: ', X_train.shape)
    # 重新获取X_test
    X_test = result[X_train.shape[0]:X_train.shape[0] + y_train.shape[0]]
    print('X_test.shape: ', X_test.shape)

    # 过采样+欠采样
    X_train, y_train = smote(X_train=X_train, y_train=y_train)

    return X_train, y_train, X_test, testing_group


def svm_model(X_train, y_train, X_test, testing_group, **params):
    """
    SVM 模型
    :param X_train:
    :param y_train:
    :param X_test:
    :param params:
    :return:
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    # model = SVC(C=5, probability=True)
    # param_grid = {
    #               'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #               'gamma': [0.001, 0.0001],
    #               'kernel': ['linear', 'rbf']}
    # grid_search = GridSearchCV(model, param_grid, n_jobs=12, verbose=1)
    # grid_search.fit(X_train, y_train)
    #
    # # 最佳参数
    # best_parameters = grid_search.best_estimator_.get_params()
    # for para, val in list(best_parameters.items()):
    #     print(para, val)
    #
    # # 模型
    # model = SVC(kernel=best_parameters['kernel'], C=best_parameters['C'], gamma=best_parameters['gamma'],
    #             probability=True)

    model = SVC(kernel='rbf', C=10, gamma=0.001, probability=True)
    model.fit(X_train, y_train)

    # 预测
    prediction = model.predict(X_test)

    # 保存结果
    save_result(testing_group=testing_group, prediction=prediction)

    return prediction


def lgb_model(X_train, y_train, X_test, testing_group, **params):
    """
    lgb 模型
    :param new_train:
    :param y:
    :param new_test:
    :param columns:
    :param params:
    :return:
    """
    import gc
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    import pandas as pd
    import lightgbm as lgb
    from sklearn.metrics import log_loss, accuracy_score

    # 线下验证
    oof = np.zeros((X_train.shape[0], 4))
    # 线上结论
    prediction = np.zeros((X_test.shape[0], 4))
    seeds = [2255, 2266, 223344, 2019 * 2 + 1024, 332232111, 40]
    num_model_seed = 5
    print('training')
    for model_seed in range(num_model_seed):
        print('模型', model_seed + 1, '开始训练')
        oof_lgb = np.zeros((X_train.shape[0], 4))
        prediction_lgb = np.zeros((X_test.shape[0], 4))
        skf = StratifiedKFold(n_splits=5, random_state=seeds[model_seed], shuffle=True)

        for index, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
            print(index)
            train_x, test_x, train_y, test_y = X_train.iloc[train_index], X_train.iloc[test_index], y_train.iloc[
                train_index], y_train.iloc[test_index]
            train_data = lgb.Dataset(train_x, label=train_y)
            validation_data = lgb.Dataset(test_x, label=test_y)
            gc.collect()
            params = {
                'learning_rate': 0.01,
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class': 4,
                'metrix': 'multi_logloss',
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'num_leaves': 100,
                'verbose': -1,
                'max_depth': 7,
                'seed': 42,
            }
            bst = lgb.train(params, train_data, valid_sets=[validation_data], num_boost_round=10000,
                            verbose_eval=1000,
                            early_stopping_rounds=1000)
            oof_lgb[test_index] += bst.predict(test_x)
            prediction_lgb += bst.predict(X_test) / 5
            gc.collect()

        oof += oof_lgb / num_model_seed
        prediction += prediction_lgb / num_model_seed
        print('logloss', log_loss(pd.get_dummies(y_train).values, oof_lgb))
        print('ac', accuracy_score(y_train, np.argmax(oof_lgb, axis=1)))

    print('logloss', log_loss(pd.get_dummies(y_train).values, oof))
    print('ac', accuracy_score(y_train, np.argmax(oof, axis=1)))

    sub = pd.DataFrame(data=testing_group.astype(int), columns=['Group'])
    prob_cols = ['Excellent ratio', 'Good ratio', 'Pass ratio', 'Fail ratio']
    for i, f in enumerate(prob_cols):
        sub[f] = prediction[:, i]
    for i in prob_cols:
        sub[i] = sub.groupby('Group')[i].transform('mean')
    sub = sub.drop_duplicates()

    sub.to_csv(path_or_buf=DefaultConfig.project_path + '/data/submit/' + DefaultConfig.select_model + '_submit.csv',
               index=False, encoding='utf-8')
    return sub


def save_result(testing_group, prediction, **params):
    """
    保存结果
    :param testing_group:
    :param prediction:
    :param params:
    :return:
    """
    import pandas as pd

    df = pd.concat([testing_group, pd.Series(prediction)], axis=1, ignore_index=True)
    df.columns = ['Group', 'predictions']

    result = pd.DataFrame(columns=['Group', 'Excellent ratio', 'Good ratio', 'Pass ratio', 'Fail ratio'])
    for (group, value) in df.groupby(['Group']):
        fail_ratio = value[value['predictions'] == 0].shape[0] / value.shape[0]
        pass_ratio = value[value['predictions'] == 1].shape[0] / value.shape[0]
        good_ratio = value[value['predictions'] == 2].shape[0] / value.shape[0]
        excellent_ratio = value[value['predictions'] == 3].shape[0] / value.shape[0]
        result.loc[result.shape[0]] = [int(group), excellent_ratio, good_ratio, pass_ratio, fail_ratio]

    result['Group'] = result['Group'].astype(int)
    result.to_csv(path_or_buf=DefaultConfig.project_path + '/data/submit/' + DefaultConfig.select_model + '_submit.csv',
                  index=False, encoding='utf-8')
