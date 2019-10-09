import pandas as pd


class Utils:

    @staticmethod
    def reduce_mem_usage(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
        """
        减少内存消耗
        :param df:
        :param verbose:
        :return:
        """
        import numpy as np

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

    @staticmethod
    def get_validation_sample_weight(X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train, y_valid,
                                     label_column: str, categorical_columns: list = 'auto', random_state: int = 42,
                                     **params) -> (pd.DataFrame, pd.DataFrame, list, list):
        """
        获取样本权重
        :param X_train:
        :param X_valid:
        :param y_train:
        :param y_valid:
        :param categorical_columns:
        :param label_column:
        :param random_state:
        :param params:
        :return:
        """
        import pandas as pd
        import lightgbm as lgb

        X_train[label_column] = list(y_train)
        X_valid[label_column] = list(y_valid)
        X_train['Is_Test'] = 0
        X_valid['Is_Test'] = 1

        # 将 Train 和 Test 合成一个数据集。Quality_label是数据本来的Y，所以剔除。
        df_adv = pd.concat([X_train, X_valid])

        adv_data = lgb.Dataset(data=df_adv.drop('Is_Test', axis=1), label=df_adv.loc[:, 'Is_Test'])

        # 定义模型参数
        params = {
            'boosting_type': 'gbdt',
            'colsample_bytree': 1,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_samples': 100,
            'min_child_weight': 1,
            'num_leaves': 36,
            'objective': 'binary',
            'random_state': random_state,
            'subsample': 1.0,
            'subsample_freq': 0,
            'metric': 'auc',
            'num_threads': 10,
            'boost_from_average': False,
            'verbose': -1
        }

        # 交叉验证
        adv_cv_results = lgb.cv(
            params,
            adv_data,
            num_boost_round=1000,
            nfold=5,
            verbose_eval=False,
            categorical_feature=categorical_columns,
            early_stopping_rounds=200,
            seed=random_state
        )

        print('交叉验证中最优的AUC为 {:.5f}，对应的标准差为{:.5f}.'.format(
            adv_cv_results['auc-mean'][-1], adv_cv_results['auc-stdv'][-1]))

        print('模型最优的迭代次数为{}.'.format(len(adv_cv_results['auc-mean'])))

        params['n_estimators'] = len(adv_cv_results['auc-mean'])

        model_adv = lgb.LGBMClassifier(**params)
        model_adv.fit(df_adv.drop('Is_Test', axis=1), df_adv.loc[:, 'Is_Test'])

        preds_adv = model_adv.predict_proba(df_adv.drop('Is_Test', axis=1))[:, 1]

        del X_train['Is_Test']
        del X_valid['Is_Test']

        # #################################### 效果不好
        # X_train['weight'] = preds_adv[:len(X_train)]
        # X_valid['weight'] = preds_adv[len(X_train):]
        #
        # X = pd.concat([X_train, X_valid], axis=0, ignore_index=True)
        # X.sort_values(by='weight', inplace=True, ascending=True)
        # X.reset_index(drop=True, inplace=True)
        #
        # return X.drop('weight', axis=1).loc[len(X_valid):, :], X.drop('weight', axis=1).loc[:len(X_valid), :], X.loc[len(
        #     X_valid):, 'weight'], X.loc[:len(X_valid), 'weight']

        # #################################### 不排序
        return X_train, X_valid, preds_adv[:len(X_train)], preds_adv[len(X_train):]
