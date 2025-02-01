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