from config import DefaultConfig

import os
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer
from sklearn import preprocessing
from collections import Counter
from imblearn.over_sampling import SMOTE
import gc
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import log_loss, accuracy_score


def get_first_round_tesing_data(**params):
    """
    返回测试集数据
    :param params:
    :return:
    """
    first_round_testing_data = pd.read_csv(DefaultConfig.first_round_testing_data_path)

    return first_round_testing_data


def get_first_round_training_data(**params):
    """
    返回训练集
    :param params:
    :return:
    """

    first_round_training_data = pd.read_csv(DefaultConfig.first_round_training_data_path)

    return first_round_training_data


def reduce_mem_usage(df, verbose=True):
    """
    减少内存消耗
    :param df:
    :param verbose:
    :return:
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


def add_feature(df, X_train, y_train, save=True, **params):
    """
    添加新的数值和类别特征
    :param df:
    :param params:
    :return:
    """
    df = df[DefaultConfig.original_columns]

    lgb_path = DefaultConfig.df_add_feature_lgb_cache_path
    cbt_path = DefaultConfig.df_add_feature_xgb_cache_path

    if (os.path.exists(lgb_path) and DefaultConfig.no_replace_add_feature) or (
            os.path.exists(cbt_path) and DefaultConfig.no_replace_add_feature):
        if DefaultConfig.select_model is 'lgb':
            df = reduce_mem_usage(pd.read_hdf(path_or_buf=lgb_path, key='add_feature', mode='r'))
        elif DefaultConfig.select_model is 'cbt':
            df = reduce_mem_usage(pd.read_hdf(path_or_buf=cbt_path, key='add_feature', mode='r'))
    else:
        # 添加新的类别列
        for column in DefaultConfig.encoder_columns:
            df[column + '_label'] = df[column].apply(lambda x: int(str(round(x))))

        # ###########################################  添加数值列
        # 生成的特征数
        n_components = 2
        generations = 5

        function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min']

        gp = SymbolicTransformer(generations=generations, population_size=2000,
                                 hall_of_fame=100, n_components=n_components,
                                 function_set=function_set,
                                 parsimony_coefficient=0.0005,
                                 max_samples=0.9, verbose=1,
                                 random_state=0, metric='spearman', n_jobs=10)

        gp.fit(X=X_train[DefaultConfig.outlier_columns], y=y_train)
        gp_features = gp.transform(df[DefaultConfig.outlier_columns])

        columns = list(df.columns)
        for i in range(n_components):
            columns.append(str(gp._best_programs[i]))

        df = pd.DataFrame(data=np.hstack((df.values, gp_features)), columns=columns, index=None)

        # ###########################################  添加类别列
        # 类别列
        # for column_i in ['Parameter7']:
        #     column_i_label = column_i + '_label'
        #     df[column_i_label] = df[column_i].apply(lambda x: int(round(x)))
        #     # 数值列
        #     for column_j in DefaultConfig.outlier_columns:
        #         stats = df.groupby(column_i)[column_j].agg(['mean', 'max', 'min'])
        #         stats.columns = ['mean_' + column_j, 'max_' + column_j, 'min_' + column_j]
        #         df = df.merge(stats, left_on=column_i, right_index=True, how='left')
        #     del df[column_i_label]

        if save:
            df.to_hdf(path_or_buf=cbt_path, key='add_feature')
    return df


def convert(df, save=True, **params):
    """
    归一化
    :param df:
    :param params:
    :return:
    """
    path = DefaultConfig.df_convert_cache_path

    if os.path.exists(path) and DefaultConfig.no_replace_convert:
        df = reduce_mem_usage(pd.read_hdf(path_or_buf=path, key='convert', mode='r'))
    else:
        max_limit = [3.9e+09, 1.4e+09, 2.9e+09, 3.7e+08, 70, 43, 2.4e+04, 7.6e+04, 6.1e+08, 1.5e+04]
        params = ['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4', 'Parameter5', 'Parameter6', 'Parameter7',
                  'Parameter8', 'Parameter9', 'Parameter10']

        max_limit_params = dict(zip(params, max_limit))

        # 处理异常值
        for column in DefaultConfig.outlier_columns:
            tmp = max_limit_params[column]
            df[column] = df[column].apply(lambda x: tmp if x > tmp else x)

            # 99.9%分位数 效果不太好
            # up_limit = np.percentile(df[column].values, 99.99)
            # # 0.1%分位数
            # low_limit = np.percentile(df[column].values, 0.01)
            # df.loc[df[column] > up_limit, column] = up_limit
            # df.loc[df[column] < low_limit, column] = low_limit

        # 获取要进行yeo-johnson变换的特征列
        columns = []
        for column in list(df.columns):
            if column not in DefaultConfig.encoder_columns and column not in DefaultConfig.label_columns:
                columns.append(column)

        print('进行yeo-johnson的特征列：')
        print(list(columns))
        # yeo-johnson 变换处理
        pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)
        df[columns] = pt.fit_transform(df[columns])

        if save:
            df.to_hdf(path_or_buf=path, key='convert')

    return df


def smote(X_train, y_train, save=True, **params):
    """
    过采样+欠采样
    :param X_train:
    :param y_train:
    :param params:
    :return:
    """
    path1 = DefaultConfig.X_train_smote_cache_path
    path2 = DefaultConfig.y_train_smote_cache_path

    if os.path.exists(path1) and os.path.exists(path2) and DefaultConfig.no_replace_smote:
        X_train = reduce_mem_usage(pd.read_hdf(path_or_buf=path1, key='X_train', mode='r'))
        y_train = pd.read_hdf(path_or_buf=path2, key='y_train', mode='r')
    else:
        # smote 算法
        smote = SMOTE(ratio={0: 1300, 1: 1600, 2: 3000, 3: 1000}, n_jobs=10)
        train_X, train_y = smote.fit_sample(X_train, y_train)
        print('Resampled dataset shape %s' % Counter(train_y))
        X_train = pd.DataFrame(data=train_X, columns=X_train.columns)
        y_train = pd.Series(train_y)

        if save:
            X_train.to_hdf(path_or_buf=path1, key='X_train')
            y_train.to_hdf(path_or_buf=path2, key='y_train')

    return X_train, y_train


def preprocess(**params):
    """
    数据预处理
    :param params:
    :return:
    """
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

    # # 一、
    # 处理类别变量 提升幅度在0.003左右
    result = add_feature(pd.concat([X_train, X_test], axis=0, ignore_index=True), X_train, y_train)
    # 去除index
    result.reset_index(inplace=True, drop=True)

    # 二、
    # 分布变换
    result = convert(result)
    # 去除index
    result.reset_index(inplace=True, drop=True)
    # 重新获取X_train
    X_train = result[:X_train.shape[0]]
    print('X_train.shape: ', X_train.shape)
    # 重新获取X_test
    X_test = result[X_train.shape[0]:X_train.shape[0] + y_train.shape[0]]
    print('X_test.shape: ', X_test.shape)

    # 三、
    # 过采样+欠采样
    X_train, y_train = smote(X_train=X_train, y_train=y_train)

    # 过采样后整型数据会变成浮点型数据
    for column in X_train.columns:
        if column not in DefaultConfig.original_columns:
            X_train[column] = X_train[column].astype(int)
            X_test[column] = X_test[column].astype(int)

    return X_train, y_train, X_test, testing_group


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
    if DefaultConfig.single_model:
        params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclassova',
            'num_class': 4,
            'metric': 'multi_error',
            'num_leaves': 24,
            'learning_rate': 0.005,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_seed': 0,
            'bagging_freq': 1,
            'verbose': -1,
            'reg_alpha': 1,
            'reg_lambda': 2,
            'lambda_l1': 0,
            'lambda_l2': 1,
            'num_threads': 10,
        }
        lgb_train = lgb.Dataset(X_train, y_train)
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=1300,
                        valid_sets=[lgb_train],
                        valid_names=['train'],
                        verbose_eval=100,
                        )
        prediction = gbm.predict(X_test, num_iteration=1300)

    else:
        # 线下验证
        oof = np.zeros((X_train.shape[0], 4))
        # 线上结论
        prediction = np.zeros((X_test.shape[0], 4))
        seeds = [42, 2019, 223344, 2019 * 2 + 1024, 332232111]
        num_model_seed = 1
        n_splits = 10
        print('training')
        for model_seed in range(num_model_seed):
            print('模型', model_seed + 1, '开始训练')
            oof_lgb = np.zeros((X_train.shape[0], 4))
            prediction_lgb = np.zeros((X_test.shape[0], 4))
            skf = StratifiedKFold(n_splits=n_splits, random_state=seeds[model_seed], shuffle=True)

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
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'num_leaves': 100,
                    'verbose': -1,
                    'max_depth': 7,
                    'seed': 42
                }
                bst = lgb.train(params, train_data, valid_sets=[validation_data], num_boost_round=10000,
                                verbose_eval=1000, early_stopping_rounds=1000)
                oof_lgb[test_index] += bst.predict(test_x, num_iteration=1300)
                prediction_lgb += bst.predict(X_test, num_iteration=1300) / n_splits
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


def cbt_model(X_train, y_train, X_test, testing_group, **params):
    """
    catboost_model
    :param X_train:
    :param y_train:
    :param X_test:
    :param columns:
    :param params:
    :return:
    """
    import gc
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    import pandas as pd
    import catboost as cbt
    from sklearn.metrics import log_loss, roc_auc_score

    print(X_train.shape, X_test.shape)
    oof = np.zeros((X_train.shape[0], 4))
    prediction = np.zeros((X_test.shape[0], 4))
    seeds = [42, 2019, 2019 * 2 + 1024, 4096, 2048, 1024]
    num_model_seed = 1
    for model_seed in range(num_model_seed):
        print(model_seed + 1)
        oof_cat = np.zeros((X_train.shape[0], 4))
        prediction_cat = np.zeros((X_test.shape[0], 4))
        skf = StratifiedKFold(n_splits=5, random_state=seeds[model_seed], shuffle=True)
        for index, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
            print(index)
            train_x, test_x, train_y, test_y = X_train.iloc[train_index], X_train.iloc[test_index], y_train.iloc[
                train_index], y_train.iloc[test_index]
            gc.collect()
            cbt_model = cbt.CatBoostClassifier(iterations=10000, learning_rate=0.001, verbose=300, max_depth=7,
                                               early_stopping_rounds=2019, task_type='GPU',
                                               loss_function='MultiClass')
            cbt_model.fit(train_x, train_y, eval_set=(test_x, test_y))
            oof_cat[test_index] += cbt_model.predict_proba(test_x)
            prediction_cat += cbt_model.predict_proba(X_test) / 5
            gc.collect()
        oof += oof_cat / num_model_seed
        prediction += prediction_cat / num_model_seed
        print('logloss', log_loss(pd.get_dummies(y_train).values, oof_cat))
        print('ac', accuracy_score(y_train, np.argmax(oof_cat, axis=1)))
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


def merge(**params):
    """
    merge
    :param params:
    :return:
    """
    lgb_submit = pd.read_csv(filepath_or_buffer=DefaultConfig.lgb_submit_path)
    cbt_submit = pd.read_csv(filepath_or_buffer=DefaultConfig.cbt_submit_path)

    lgb_submit['sum'] = 0
    for column in ['Excellent ratio', 'Good ratio', 'Pass ratio', 'Fail ratio']:
        lgb_submit[column] = (0.7 * lgb_submit[column] + 0.3 * cbt_submit[column])
        lgb_submit['sum'] += lgb_submit[column]

    for column in ['Excellent ratio', 'Good ratio', 'Pass ratio', 'Fail ratio']:
        lgb_submit[column] /= lgb_submit['sum']

    del lgb_submit['sum']

    lgb_submit.to_csv(DefaultConfig.submit_path, encoding='utf-8', index=None)
