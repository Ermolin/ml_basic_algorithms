from sklearn.datasets import make_regression
import pandas as pd
import numpy as np

class MyLineReg():
    def __init__(
            self,
            n_iter: int = 10,
            learning_rate: float = 0.5,
            metric: str = None,
            weights: np.array = None,
            metric_value: float = None
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.metric_value = metric_value

    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def MSE(
            self,
            y: np.array, # Series with real values
            y_cap : np.array, # Series with predicted values
            ):
        '''
        calculate MSE
        '''
        errors = y-y_cap
        return sum([float(i)**2 for i in errors])/len(errors)
         
    def MAE(
            self,
            y: np.array, # Series with real values
            y_cap : np.array, # Series with predicted values
            ):
        '''
        calculate MAE
        '''
        errors = y-y_cap
        return sum([np.abs(i) for i in errors])/len(errors)

    def RMSE(
            self,
            y: np.array, # Series with real values
            y_cap : np.array, # Series with predicted values
            ):
        '''
        calculate RMSE
        '''
        errors = y-y_cap
        return (sum([float(i)**2 for i in errors])/len(errors))** (1/2)
    
    def R2(
            self,
            y: np.array, # Series with real values
            y_cap : np.array, # Series with predicted values
            ):
        '''
        calculate R^2
        '''
        errors = y-y_cap
        y_mean = np.mean(y)
        errors2 = y - y_mean
        return 1 - (sum([float(i)**2 for i in errors]) / sum([float(i)**2 for i in errors2]))

    def MAPE(
            self,
            y: np.array, # Series with real values
            y_cap : np.array, # Series with predicted values
            ):
        '''
        calculate MAPE
        '''
        errors = y-y_cap
        part_errors = np.abs(errors / y)
        return 100 * sum(part_errors)/len(y)

    def ANTIGRADIENT(
            self,
            y: np.array, # Series with real values
            y_cap : np.array, # Series with predicted values
            X: np.array, # df with features
        ):
        '''
        calculate antigradient
        '''
        errors = y_cap-y
        errors_df = errors * X.transpose()
        antigradient = - (errors_df.sum(axis=1) / X.shape[0] )*2
        return antigradient

    def calc_error(
            self,
            y: pd.Series, # Series with real values
            y_cap : pd.Series, # Series with predicted values
            ):
        '''
        calculate selected metric
        '''
        if self.metric == 'mse':
            return self.MSE(y=y,
                            y_cap=y_cap)
        elif self.metric == 'mae':
            return self.MAE(y=y,
                            y_cap=y_cap)
        elif self.metric == 'rmse':
            return self.RMSE(y=y,
                            y_cap=y_cap)
        elif self.metric == 'r2':
            return self.R2(y=y,
                            y_cap=y_cap)
        elif self.metric == 'mape':
            return self.MAPE(y=y,
                            y_cap=y_cap)
        else:
            return None

    def fit(
            self,
            X: np.array, # df with features
            y: np.array, # Series with real values
            verbose: int = None
            ):
        '''
        fit the model'''

        # initiating weights array
        W = np.ones(len(list(X.columns))+1)
        # add w0 for X0 as array of 1
        # change types to np.array
        feats = list(X.columns)
        X['x0']=1
        X=np.array(X[['x0']+feats])
        y= np.array(y)


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

            self.metric_value = self.calc_error(y=y,y_cap=y_cap)

            if verbose and self.metric :
                if i == 0:
                    print(f'start | loss: {mse} | {self.metric} : {self.metric_value}') 
                
                elif (i+1) % verbose == 0:
                    print(f'{i} | loss: {mse} | {self.metric} : {self.metric_value}') 
        
        # update metric after all changes
        X_vals =X*W
        y_cap = X_vals.sum(axis=1)
        self.metric_value = self.calc_error(y=y,y_cap=y_cap)

    def get_coef(self):
        '''
        return weights except w0 representing 
        '''
        return np.array(self.weights[1:])

    def predict(
                self,
                X: pd.DataFrame, # df with features
                ):
        W = self.weights
        feats = list(X.columns)
        X['x0']=1
        X=X[['x0']+feats]
        X=np.array(X)
        return pd.Series((X*W).sum(axis=1))
    
    def get_best_score(
                       self
                      ):
        return self.metric_value
