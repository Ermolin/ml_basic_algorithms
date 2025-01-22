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
            random_state: int = None,
            sgd_sample: Union[float,int]= None,
            reg: str = None,
            l1_coef: float = None,
            l2_coef: float = None,
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.random_state = random_state
        self.sgd_sample = sgd_sample
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

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
        antigradient = -( (errors_df.sum(axis=1) / X.shape[0] )) #- self.REGULARISATION(W=W)
        return antigradient

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

            #self.metric_value = self.calc_error(y=y,y_cap=y_cap)

            # if verbose : #and self.metric :
            #     if i == 0:
            #         print(f'start | loss: {logloss} | {self.metric} : {self.metric_value}') 
                
            #     elif (i+1) % verbose == 0:
            #         print(f'{i} | loss: {logloss} | {self.metric} : {self.metric_value}') 
        
    def get_coef(self):
        '''
        return weights except w0 representing 
        '''
        return np.array(self.weights[1:]) 