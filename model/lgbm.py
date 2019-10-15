import gc
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

from utils.utils import Utils
from sklearn.model_selection import StratifiedKFold
from configuration.config import DefaultConfig


class LightGbm(object):
    def __init__(self, X_train, y_train, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test

    def main(self, testing_group, **params):
        """
        lgb 模型
        :param testing_group:
        :param params:
        :return:
        """
        feature_importance = None
        # 线下验证
        oof = np.zeros((self.X_train.shape[0], 4))
        # 线上结论
        prediction = np.zeros((self.X_test.shape[0], 4))
        seeds = [42, 2019, 223344, 2019 * 2 + 1024, 332232111]
        num_model_seed = 1
        n_splits = 5
        print('training')

        params = {
            'learning_rate': 0.01,
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class': 4,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 7,
            'seed': 42,
            'num_leaves': 1000
        }

        # 寻找最优的num_leaves
        # min_merror = np.inf
        # for num_leaves in [150, 200, 250, 300, 500, 1000]:
        #     params["num_leaves"] = num_leaves
        #
        #     cv_results = lgb.cv(params=params,
        #                         train_set=lgb.Dataset(X_train, label=y_train),
        #                         num_boost_round=2000,
        #                         stratified=False,
        #                         nfold=5,
        #                         verbose_eval=50,
        #                         seed=23,
        #                         early_stopping_rounds=20)
        #
        #     mean_error = min(cv_results['multi_logloss-mean'])
        #
        #     if mean_error < min_merror:
        #         min_merror = mean_error
        #         params["num_leaves"] = num_leaves
        #
        # print('num_leaves: ', num_leaves)

        for model_seed in range(num_model_seed):
            print('模型', model_seed + 1, '开始训练')
            oof_lgb = np.zeros((self.X_train.shape[0], 4))
            prediction_lgb = np.zeros((self.X_test.shape[0], 4))
            skf = StratifiedKFold(n_splits=n_splits, random_state=seeds[model_seed], shuffle=True)

            # 存放特征重要性
            feature_importance_df = pd.DataFrame()
            for index, (train_index, test_index) in enumerate(skf.split(self.X_train, self.y_train)):
                print(index)
                train_x, test_x, train_y, test_y = self.X_train.iloc[train_index], self.X_train.iloc[test_index], \
                                                   self.y_train.iloc[train_index], self.y_train.iloc[test_index]

                # train_data, validation_data, train_data_weight, validation_data_weight = get_validation(train_x, test_x,
                #                                                                                         train_y, test_y,
                #                                                                                         ['Parameter10',
                #                                                                                          'Parameter5',
                #                                                                                          'Parameter6',
                #                                                                                          'Parameter9',
                #                                                                                          'Parameter8',
                #                                                                                          'Parameter7',
                #                                                                                          'Parameter10_label'],
                #                                                                                         seeds[model_seed])
                #
                # train_data = lgb.Dataset(train_data.drop('Quality_label', axis=1), label=train_data.loc[:, 'Quality_label'],
                #                          weight=train_data_weight)
                # validation_data = lgb.Dataset(validation_data.drop('Quality_label', axis=1),
                #                               label=validation_data.loc[:, 'Quality_label'],
                #                               weight=validation_data_weight)

                train_data = lgb.Dataset(train_x, label=train_y)
                validation_data = lgb.Dataset(test_x, label=test_y)

                gc.collect()
                bst = lgb.train(params, train_data, valid_sets=[validation_data], num_boost_round=10000,
                                verbose_eval=1000, early_stopping_rounds=1000)
                oof_lgb[test_index] += bst.predict(test_x, num_iteration=1300)
                prediction_lgb += bst.predict(self.X_test, num_iteration=1300) / n_splits
                gc.collect()

                fold_importance_df = pd.DataFrame()
                fold_importance_df["feature"] = list(self.X_train.columns)
                fold_importance_df["importance"] = bst.feature_importance(importance_type='split',
                                                                          iteration=bst.best_iteration)
                fold_importance_df["fold"] = index + 1
                feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            oof += oof_lgb / num_model_seed
            prediction += prediction_lgb / num_model_seed
            print('logloss', log_loss(pd.get_dummies(self.y_train).values, oof_lgb))
            print('ac', accuracy_score(self.y_train, np.argmax(oof_lgb, axis=1)))

            if feature_importance is None:
                feature_importance = feature_importance_df
            else:
                feature_importance += feature_importance_df

        feature_importance['importance'] /= num_model_seed
        print('logloss', log_loss(pd.get_dummies(self.y_train).values, oof))
        print('ac', accuracy_score(self.y_train, np.argmax(oof, axis=1)))

        if feature_importance is not None:
            feature_importance.to_hdf(path_or_buf=DefaultConfig.lgb_feature_cache_path, key='lgb')
            # 读取feature_importance_df
            feature_importance_df = Utils.reduce_mem_usage(
                pd.read_hdf(path_or_buf=DefaultConfig.lgb_feature_cache_path, key='lgb', mode='r'))

            plt.figure(figsize=(8, 8))
            # 按照flod分组
            group = feature_importance_df.groupby(by=['fold'])

            result = []
            for key, value in group:
                value = value[['feature', 'importance']]

                result.append(value)

            result = pd.concat(result)
            print(result.groupby(['feature'])['importance'].agg('mean').sort_values(ascending=False).head(40))
            # 5折数据取平均值
            result.groupby(['feature'])['importance'].agg('mean').sort_values(ascending=False).head(40).plot.barh()
            plt.show()

        sub = pd.DataFrame(data=testing_group.astype(int), columns=['Group'])
        for i, f in enumerate(DefaultConfig.columns):
            sub[f] = prediction[:, i]
        for i in DefaultConfig.columns:
            sub[i] = sub.groupby('Group')[i].transform('mean')
        sub = sub.drop_duplicates()

        sub.to_csv(
            path_or_buf=DefaultConfig.project_path + '/data/submit/' + DefaultConfig.select_model + '_submit.csv',
            index=False, encoding='utf-8')
        return sub
