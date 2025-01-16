class MyLineReg():
    def __init__(
            self,
            
            n_iter: int = 10,
            learning_rate: float = 0.5
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'