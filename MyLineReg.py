from sklearn.datasets import make_regression
import pandas as pd
import numpy as np

class MyLineReg():
    def __init__(
            self,
            n_iter: int = 10,
            learning_rate: float = 0.5,
            weights = None
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def MSE(
            self,
            y: pd.Series, # Series with real values
            y_cap : pd.Series, # Series with predicted values
            ):
        '''
        calculate MSE
        '''
        errors = y-y_cap
        mse = sum([float(i)**2 for i in errors])/len(errors)
        return mse

    def ANTIGRADIENT(
            self,
            y: pd.Series, # Series with real values
            y_cap : pd.Series, # Series with predicted values
            X: pd.DataFrame, # df with features
        ):
        '''
        calculate antigradient
        '''
        errors = y_cap-y
        errors_df = errors * X.transpose()
        antigradient = - (errors_df.sum(axis=1) / X.shape[0] )*2
        return antigradient

    def fit(
            self,
            X: pd.DataFrame, # df with features
            y: pd.Series, # Series with real values
            verbose: int = None
            ):
        '''
        fit the model'''
        # add w0 for X0 as array of 1
        feats = list(X.columns)
        X['x0']=1
        X=X[['x0']+feats]

        # initiating weights array
        W = np.ones(len(list(X.columns)))

        for i in range(self.n_iter):
            X_vals =X*W
            y_cap = X_vals.sum(axis=1)

            mse = self.MSE(y=y,
                      y_cap=y_cap)

            antigradient = self.ANTIGRADIENT(y=y,
                                        y_cap=y_cap,
                                        X=X)

            W = W + (antigradient* self.learning_rate)
            self.weights = W

            if verbose:
                if i == 0:
                    print(f'start | loss: {mse}') 
                
                elif (i+1) % verbose == 0:
                    print(f'{i} | loss: {mse}')

    def get_coef(self):
        '''
        return weights except w0 representing 
        '''
        return np.array(self.weights[1:])
