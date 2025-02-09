import pandas as pd
import numpy as np


import random
from typing import Callable, Union, List

class MyLogReg():
    def __init__(
            self,
            n_iter: int = 10,
            learning_rate: Union[float, Callable[[float],float]] = 0.1,
            weights: np.array = None,
            random_state: int = 42,
            sgd_sample: Union[float,int]= None,
            reg: str = None,
            l1_coef: float = None,
            l2_coef: float = None,
            metric: str = None,
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.random_state = random_state
        self.sgd_sample = sgd_sample
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.metric = metric
        self.best_score_val = None

    def __str__(self):
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def SGS(self,
            X: np.array, # df with features
            #y: np.array, # Series with real values
            ): #Stochastic Gradient Sample
        
        if isinstance(self.sgd_sample, float):
            sample_size = int(X.shape[0]*self.sgd_sample)
        elif isinstance(self.sgd_sample, int):
            sample_size = self.sgd_sample
        else:
            sample_size = X.shape[0]
        
        sample_rows_idx = random.sample(range(X.shape[0]),sample_size)

        return sample_rows_idx

    def LogLoss(
            self,
            y: np.array, # Series with real values
            y_cap : np.array, # Series with predicted values
            ):
        '''
        calculate LogLoss
        '''

        return -1/y.shape[0] * (np.sum( (y* np.log(y_cap+1e-15 ))) + np.sum(((1-y)*np.log(1-y_cap+1e-15 ) ))) 
               #1e-15 small number to avoid -inf in log results
    
    def REGULARISATION(
            self,
            W: np.array,
    ):
        '''
        calculate regularisation
        '''
        if self.reg == 'l1':
            return np.sign(W) * self.l1_coef
        elif self.reg == 'l2':
            return (W * 2)  * self.l2_coef 
        elif self.reg == 'elasticnet':
            return (np.sign(W) * self.l1_coef) + ((W * 2)  * self.l2_coef )
        else:
            return np.zeros(len(W))

    def ANTIGRADIENT(
            self,
            y: np.array, # Series with real values
            y_cap : np.array, # Series with predicted values
            X: np.array, # df with features
            W: np.array,
        ):
        '''
        calculate antigradient
        '''
        errors = y_cap-y
        errors_df = errors * X.transpose()
        antigradient = -( (errors_df.sum(axis=1) / X.shape[0] )) - self.REGULARISATION(W=W)
        return antigradient

    def get_class_score(
            self,
            y: np.array,
            y_cap: np.array,
            ):
        TP = np.sum((y==1) & (y_cap==1))
        TN = np.sum((y==0) & (y_cap==0))
        FP = np.sum((y==0) & (y_cap==1))
        FN = np.sum((y==1) & (y_cap==0))
        return TP, TN, FP, FN

    def best_score(
            self,
            X: np.array, # df with features
            y: np.array, # Series with real values
            ):
        ''' get last score by choosen metric'''
        y_cap = self.predict(X=X)
        TP, TN, FP, FN = self.get_class_score(y=y, y_cap=y_cap)

        if self.metric == 'accuracy':

            self.best_score_val= (TP+TN)/(TP+TN+FP+FN)
        elif self.metric == 'precision':
            self.best_score_val= TP/(TP+FP)
        elif self.metric == 'recall':
            self.best_score_val= TP/(TP+FN)
        elif self.metric == 'f1':
            self.best_score_val= 2*TP/(2*TP+FP+FN)
        elif self.metric == 'roc_auc':
            y_score = self.predict_proba(X=X)
            sorted_indices = np.argsort(-y_score)
            y_true_sorted = y[sorted_indices]

            pos = np.sum(y)
            neg = len(y) - pos

            cum_pos = np.cumsum(y_true_sorted)

            self.best_score_val= np.sum(cum_pos[y_true_sorted == 0]) / (pos * neg)
        else:
            self.best_score_val= None

    def get_best_score(self):
        return self.best_score_val

    def fit(
            self,
            X: np.array, # df with features
            y: np.array, # Series with real values
            verbose: int = None
            ):
        '''
        fit the model'''
        # fix random seed
        random.seed(self.random_state)

        # initiating weights array
        W = np.ones(len(list(X.columns))+1)
        self.weights = W
        # add w0 for X0 as array of 1
        # change types to np.array
        X_df = X.copy()
        feats = list(X.columns)
        X['x0']=1
        X=np.array(X[['x0']+feats])
        y= np.array(y)

        # make all learning_rates lambda funcs
        if isinstance(self.learning_rate, float):
            value = self.learning_rate
            self.learning_rate = lambda x: value


        for i in range(self.n_iter):
            
            subset_indexes = self.SGS(X=X) #,y=y)

            LR = self.learning_rate(i + 1) # make start iters for lambda from 1 to N

            X_vals =X*W
            y_cap = 1 / (1+np.e ** (-X_vals.sum(axis=1)))

            logloss = self.LogLoss(y=y,
                      y_cap=y_cap)

            antigradient = self.ANTIGRADIENT(y=y[subset_indexes],
                                        y_cap=y_cap[subset_indexes],
                                        X=X[subset_indexes],
                                        W=W)

            
            W = W + (antigradient * LR)
            self.weights = W

            best_score_val = self.best_score(X=X_df, y=y)
        
    def get_coef(self):
        '''
        return weights except w0 representing 
        '''
        return np.array(self.weights[1:]) 
    
    def predict_proba(self,
                      X: pd.DataFrame):
        '''
        predict probabilities
        '''
        #prep matrix
        X_in=X.copy()
        feats = list(X_in.columns)
        X_in['x0']=1
        X_in = X_in[['x0']+feats]
        X_in=np.array(X_in)

        #X_vals =X*self.weights
        #y_cap = 1 / (1+np.e ** (-X_vals.sum(axis=1)))
        y_cap = 1 / (1 + np.exp(- self.weights @ X_in.T))
        return y_cap
    
    def predict(self,
                X: pd.DataFrame,
                threshold: float = 0.5):
        '''
        predict classes
        '''
        y_cap = self.predict_proba(X=X)
        return np.where(y_cap>threshold,1,0)