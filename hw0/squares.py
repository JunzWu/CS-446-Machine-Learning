import numpy
import torch

def numpy_squares(k):
    """return (1, 4, 9, ..., k^2) as a numpy array"""
    array = numpy.zeros(k)
    for i in range(k):
     array[i] = (i+1)**2
     return array
    pass
print(numpy_squares(5))
def torch_squares(k):
    """return (1, 4, 9, ..., k^2) as a torch array"""
    array = torch.zeros(k)
    for i in range(k):
        array[i] = (i+1)**2
    return array
    pass

