import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer


class NumericalFeature(object):
    def __init__(self, df, X_train, y_train):
        self.df = df
        self.X_train = X_train
        self.y_train = y_train

    def symbolic_transformer(self, generations, n_components, columns):
        function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min']

        gp = SymbolicTransformer(generations=generations, population_size=2000,
                                 hall_of_fame=100, n_components=n_components,
                                 function_set=function_set,
                                 parsimony_coefficient=0.0005,
                                 max_samples=0.8, verbose=1,
                                 random_state=0, metric='spearman', n_jobs=10)

        gp.fit(X=self.X_train[columns], y=self.y_train)
        gp_features = gp.transform(self.df[columns])

        columns = list(self.df.columns)
        for i in range(n_components):
            columns.append(str(gp._best_programs[i]))

        df = pd.DataFrame(data=np.hstack((self.df.values, gp_features)), columns=columns, index=None)

        return df
