import numpy as np

class MinMaxScaler:

    def __init__(self, input):

        self.min = np.min(input, axis=0)
        self.max = np.max(input, axis=0)

    def transform(self, x):

        return (x - self.min)/(self.max - self.min)
    
    def inverse_transform(self, x):
            
        return x*(self.max - self.min) + self.min
    
class MinMaxScalerExact:

    def __init__(self, min, max):

        self.min = min
        self.max = max

    def transform(self, x):
            
        return (x - self.min)/(self.max - self.min)
    
    def inverse_transform(self, x):
                
        return x*(self.max - self.min) + self.min