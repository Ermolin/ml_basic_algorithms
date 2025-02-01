import pandas as pd
import numpy as np


class MyKNNReg():
    def __init__(
            self, 
            k: int = 3,
    ):
        self.k = k 
        self.train_size: tuple[int, int] = None
        self.X: pd.DataFrame = None
        self.y: pd.Series = None

    def __str__(self):
        return f'MyKNNReg class: k={self.k}'
    
    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            ):
        self.X = X
        self.y = y 
        self.train_size = ( X.shape[0],X.shape[1])
        return None
    
    def euclidean(self,
                  arr1: np.array,
                  arr2: np.array,):
        return np.sqrt(np.sum((arr2-arr1)**2)) 
    
    def predict(self,
                X: pd.DataFrame,
                ):
        y_pred = []
        for i in range(X.shape[0]):
            dist = []
            for j in range(self.X.shape[0]):
                dist.append(self.euclidean(X.iloc[i].values,self.X.iloc[j].values))
            dist = np.array(dist)
            idx = np.argsort(dist)[:self.k]
            y_pred.append(np.mean(self.y.iloc[idx]))
        return np.array(y_pred)