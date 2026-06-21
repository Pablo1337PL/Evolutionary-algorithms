import numpy as np

class ActivationFunction:
    def __init__(self, name):
        self.name = name

    def function(self, x):
        pass

    def derivative(self, x):
        pass

class Linear(ActivationFunction):
    def __init__(self):
        super().__init__("Linear")

    def function(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class Sigmoid(ActivationFunction):
    def __init__(self):
        super().__init__("Sigmoid")

    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, a):
        return a * (1 - a)
    
class ReLU(ActivationFunction):
    def __init__(self):
        super().__init__("ReLU")

    def function(self, x):
        return np.maximum(0, x)

    def derivative(self, a):
        return (a > 0).astype(float)
    

class Tanh(ActivationFunction):
    def __init__(self):
        super().__init__("Tanh")

    def function(self, x):
        return np.tanh(x)

    def derivative(self, a):
        return 1 - a ** 2

class Softmax(ActivationFunction):
    def __init__(self):
        super().__init__("Softmax")

    def function(self, x):
        x_2d = np.atleast_2d(x)
        
        shifted_x = x_2d - np.max(x_2d, axis=1, keepdims=True)
        exps = np.exp(shifted_x)
        result = exps / np.sum(exps, axis=1, keepdims=True)
        
        if np.ndim(x) == 1:
            return result[0]
        
        return result

    def derivative(self, x):
        return np.ones_like(x) # because it's usually used with cross-entropy loss which simplifies the derivative calculation.
