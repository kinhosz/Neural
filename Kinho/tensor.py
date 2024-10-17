import numpy as np
from typing import List, Union
import warnings

class Tensor(object):
    """
    A class representing a multi-dimensional tensor.

    Args:
        shape: tuple[int]
            the tensor's size
    """

    _data: Union[List[float], List['Tensor']] = []
    _len: int = 0

    def __init__(self, shape: tuple[int], uniform=False):
        self._len = shape[0]

        if len(shape) != 1:
            for _ in range(self._len):
                self._data.append(Tensor(shape[1:], uniform=uniform))
        elif uniform:
            self._data = np.random.uniform(-1, 1, shape[0]).tolist()
        else:
            self._data = [0.0] * shape[0]

    def __getitem__(self, index: int):
        return self._data[index]

    def __setitem__(self, index: int, value: Union[float, 'Tensor']):
        if type(self._data[0]) != type(value):
            warnings.warn(f"Expected data type: {type(self._data[0])}, but received: {type(value)}")
        if isinstance(value, Tensor) and self._data[0].shape() != value.shape():
            warnings.warn(f"Expected size: {self._data[0].shape()}, but received: {value.shape()}")

        self._data[index] = value

    def shape(self):
        if not isinstance(self._data[0], Tensor):
            return (self._len,)
        else:
            return (self._len,) + self._data[0].shape()
