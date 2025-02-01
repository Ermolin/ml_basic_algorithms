import pandas as pd
import numpy as np


class MyKNNClf():
    def __init__(
            self,
            k: int = 3,
    ):
        self.k = k
        self.train_size: tuple[int, int] = None
        self.X: pd.DataFrame = None
        self.y: pd.Series = None

    def __str__(self):
        return f'MyKNNClf class: k={self.k}'

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            ):
        self.X = X
        self.y = y 
        self.train_size = ( X.shape[0],X.shape[1])
        return None

    def euclid(self,
               arr1: np.array,
               arr2: np.array,):
        return np.sqrt(np.sum((arr2-arr1)**2))    


    def predict_proba(self,
                      X: pd.DataFrame,
                      ):
        y_pred = []
        for i in range(X.shape[0]):
            dist = []
            for j in range(self.train_size[0]):
                dist.append(self.euclid(X.iloc[i,:],self.X.iloc[j,:]))
            dist = np.array(dist)
            idx = dist.argsort()  
            y_pred.append(np.mean(self.y.iloc[idx][:self.k]))
        return np.array(y_pred)
    
    def predict(self,
                X: pd.DataFrame,
                ):
        y_pred = self.predict_proba(X)
        return np.round(y_pred + 0.000000000001).astype(int)