import pandas as pd
import numpy as np


class MyKNNReg():
    def __init__(
            self, 
            k: int = 3,
            metric: str = 'euclidean',
            weight: str = 'uniform',
    ):
        self.k = k 
        self.train_size: tuple[int, int] = None
        self.X: pd.DataFrame = None
        self.y: pd.Series = None
        self.metric = metric
        self.weight = weight

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
    
    def chebyshev(self,
                  arr1: np.array,
                  arr2: np.array,):
        return np.max(np.abs(arr2-arr1))   
    
    def manhattan(self,
                  arr1: np.array,
                  arr2: np.array,):
        return np.sum(np.abs(arr2-arr1)) 

    def cosine(self,
                  arr1: np.array,
                  arr2: np.array,):
        return 1-np.dot(arr1,arr2)/(np.linalg.norm(arr1)*np.linalg.norm(arr2))
    
    def calc_metric(self,
                    arr1: np.array,
                    arr2: np.array,):
        if self.metric == 'euclidean':
            return self.euclidean(arr1,arr2)
        elif self.metric == 'chebyshev':
            return self.chebyshev(arr1,arr2)
        elif self.metric == 'manhattan':
            return self.manhattan(arr1,arr2)
        elif self.metric == 'cosine':
            return self.cosine(arr1,arr2)
        else:
            return None
    
    def predict(self,
                X: pd.DataFrame,
                ):
        
        if self.weight == 'uniform':
            y_pred = []
            for i in range(X.shape[0]):
                dist = []
                for j in range(self.X.shape[0]):
                    dist.append(self.calc_metric(X.iloc[i].values,self.X.iloc[j].values))
                dist = np.array(dist)
                idx = np.argsort(dist)[:self.k]
                y_pred.append(np.mean(self.y.iloc[idx]))
            return np.array(y_pred)

        elif self.weight == 'rank':
            y_pred = []
            for i in range(X.shape[0]):
                dist = []
                for j in range(self.X.shape[0]):
                    dist.append(self.calc_metric(X.iloc[i].values,self.X.iloc[j].values))
                dist = np.array(dist)
                idx = np.argsort(dist)
                
                y_pred_i_top = []
                y_pred_i_bot = []
                w_all = np.sum([1/(j+1) for j in range(self.k)])
                for j in range(self.k):
                    w_j = (1/(j+1)) / w_all
                    y_pred_i_top.append(self.y.iloc[idx[j]] * w_j)
                y_pred.append(np.sum(y_pred_i_top))

            return np.array(y_pred)
        
            
        elif self.weight == 'distance':
            y_pred = []
            for i in range(X.shape[0]):
                dist = []
                for j in range(self.train_size[0]):
                    dist.append(self.calc_metric(X.iloc[i,:],self.X.iloc[j,:]))
                dist = np.array(dist)
                idx = dist.argsort()

                y_pred_i_top = []
                w_all = np.sum([1/dist[idx[j]] for j in range(self.k)])
                for j in range(self.k):
                    w_j = (1/dist[idx[j]]) / w_all
                    y_pred_i_top.append(self.y.iloc[idx[j]] *w_j)
                y_pred.append(np.sum(y_pred_i_top))

            return np.array(y_pred)
        
        else:
            return None