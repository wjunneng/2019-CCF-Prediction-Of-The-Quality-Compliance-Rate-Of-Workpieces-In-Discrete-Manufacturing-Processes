import numpy as np
import catboost as cbt
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from utils.utils import Utils
from configuration.config import DefaultConfig


class CatBoost(object):
    def __init__(self, X_train, y_train, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test

    def main(self, testing_group, **params):
        """
        catboost_model
        :param testing_group:
        :param params:
        :return:
        """
        print('cbt train...')
        feature_importance = None
        oof = np.zeros((self.X_train.shape[0], 4))
        prediction = np.zeros((self.X_test.shape[0], 4))
        seeds = [42, 2019 * 2 + 1024, 4096, 2048, 1024]
        num_model_seed = 1
        n_splits = 10
        for model_seed in range(num_model_seed):
            print(model_seed + 1)
            oof_cat = np.zeros((self.X_train.shape[0], 4))
            prediction_cat = np.zeros((self.X_test.shape[0], 4))
            skf = StratifiedKFold(n_splits=n_splits, random_state=seeds[model_seed], shuffle=True)

            # 存放特征重要性
            feature_importance_df = pd.DataFrame()
            for index, (train_index, test_index) in enumerate(skf.split(self.X_train, self.y_train)):
                print(index)
                train_x, test_x, train_y, test_y = self.X_train.iloc[train_index], self.X_train.iloc[test_index], \
                                                   self.y_train.iloc[train_index], self.y_train.iloc[test_index]
                gc.collect()
                bst = cbt.CatBoostClassifier(iterations=1200, learning_rate=0.005, verbose=300,
                                             early_stopping_rounds=1000, task_type='GPU',
                                             loss_function='MultiClass')

                # train_data, validation_data, train_data_weight, validation_data_weight = get_validation(train_x, test_x,
                #                                                                                         train_y, test_y,
                #                                                                                         ['Parameter10',
                #                                                                                          'Parameter9',
                #                                                                                          'Parameter8',
                #                                                                                          'Parameter5',
                #                                                                                          'Parameter6',
                #                                                                                          'Parameter7'],
                #                                                                                         seeds[model_seed])
                #
                # bst.fit(train_data.drop('Quality_label', axis=1), train_data['Quality_label'],
                #         eval_set=(validation_data.drop('Quality_label', axis=1), validation_data['Quality_label']),
                #         sample_weight=train_data_weight)
                bst.fit(train_x, train_y,
                        eval_set=(test_x, test_y))

                oof_cat[test_index] += bst.predict_proba(test_x)
                prediction_cat += bst.predict_proba(self.X_test) / n_splits
                gc.collect()

                fold_importance_df = pd.DataFrame()
                fold_importance_df["feature"] = list(self.X_train.columns)
                fold_importance_df["importance"] = bst.get_feature_importance()
                fold_importance_df["fold"] = index + 1
                feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            oof += oof_cat / num_model_seed
            prediction += prediction_cat / num_model_seed
            print('logloss', log_loss(pd.get_dummies(self.y_train).values, oof_cat))
            print('ac', accuracy_score(self.y_train, np.argmax(oof_cat, axis=1)))
            print('mae', 1 / (1 + np.sum(np.absolute(np.eye(4)[self.y_train] - oof_cat)) / 480))

            if feature_importance is None:
                feature_importance = feature_importance_df
            else:
                feature_importance += feature_importance_df

        feature_importance['importance'] /= num_model_seed
        print('logloss', log_loss(pd.get_dummies(self.y_train).values, oof))
        print('ac', accuracy_score(self.y_train, np.argmax(oof, axis=1)))
        print('mae', 1 / (1 + np.sum(np.absolute(np.eye(4)[self.y_train] - oof)) / 480))

        if feature_importance is not None:
            feature_importance.to_hdf(path_or_buf=DefaultConfig.cbt_feature_cache_path, key='cbt')
            # 读取feature_importance_df
            feature_importance_df = Utils.reduce_mem_usage(
                pd.read_hdf(path_or_buf=DefaultConfig.cbt_feature_cache_path, key='cbt', mode='r'))
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
