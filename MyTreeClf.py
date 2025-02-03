import pandas as pd
import numpy as np


class MyTreeClf():
    def __init__(
            self,  
            max_depth: int = 5,
            min_samples_split: int = 2,
            max_leafs: int = 20, # aka max_leaves
    ):
        self.max_depth = max_depth  
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

    def __str__(self):
        return f'MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}'