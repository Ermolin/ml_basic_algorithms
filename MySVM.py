import pandas as pd
import numpy as np


import random
from typing import Callable, Union, List



class MySVM():
    def __init__(
            self,
            n_iter: int = 10,
            learning_rate: Union[float, Callable[[float],float]] = 0.001,
            weights: np.array = None,
            b: float = None,
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.b = b

    def __str__(self):
        return f'MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def svm_loss(self, X, y):
        # hinge loss
        loss = np.sum(np.maximum(0, 1 - y * (np.sum(X *  self.weights, axis=1) + self.b)))
        return loss

    def fit(
            self,
            X: pd.DataFrame, # df with features
            y: pd.DataFrame, # Series with real values
            verbose: int = None
            ):
        self.b=1.
        # initiating weights array
        self.weights = np.ones(len(list(X.columns)))
        # change types to np.array
        X=np.array(X.copy())
        y= np.array(y.copy())

        #y_metric = np.array(np.where(y > 1, 1, -1))
        y_metric = y.copy()
        y_metric[y_metric != 1] = -1
        
        # loop if we iter by sample
        for i in range(self.n_iter):
            for x_n, y_n in zip(X,y_metric):
                #mask = np.where(np.sum(np.sum(x_n*self.weights) + self.b ) >= 1, 0, 1)
                if np.sum(np.sum(x_n*self.weights) + self.b )* y_n >= 1:
                    mask = 0
                else:
                    mask = 1
                
                if mask == 1:
                    margin_grad= 2*self.weights - (y_n* x_n)
                else:
                    margin_grad = 2*self.weights
                
                if mask ==1:
                    eror_grad = - y_n
                else:
                    eror_grad = 0

                # apply antigradient
                self.weights -=(self.learning_rate * margin_grad)
                self.b -= (self.learning_rate * eror_grad) 

            if verbose and (i+1)%verbose==0:
                print(f'iteration {i+1} loss: {self.svm_loss(X, y)}, weights: {self.weights}, b: {self.b}')

    def get_coef(self):
        return (self.weights, self.b)
    
    def predict_proba(self, X: pd.DataFrame):
        X_np = np.array(X)
        return np.sign(np.sum(X_np *  self.weights, axis=1) + self.b)
    
    def predict(self, X: pd.DataFrame):
        return np.where(self.predict_proba(X) == 1, 1, 0)