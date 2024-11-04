import numpy as np
from typing import List, Union

class Tensor(object):
    """
    A class representing a multi-dimensional tensor.

    Args:
        shape: tuple[int] = None
            the tensor's size
        
    """

    def __init__(self, shape: tuple[int] = None, uniform=False):
        self._data: Union[List[float], List['Tensor']] = []
        self._len = 0

        if shape:
            self._len = shape[0]
            if len(shape) != 1:
                for _ in range(self._len):
                    self._data.append(Tensor(shape[1:], uniform=uniform))
            elif uniform:
                self._data = np.random.uniform(low=-1, high=1, size=self._len).tolist()
            else:
                self._data = [float(0.0)] * self._len

    def load_from(self, data: list):
        if not is_list(data) or len(data) == 0:
            raise TypeError(f"Expected list with len > 0, received {data}")

        if not is_list(data[0]) and not is_float(data[0]) and not is_tensor(data[0]):
            raise TypeError(f"Expected list of list or list of floats. Received list of {type(data[0])}")

        self._len = len(data)

        if is_list(data[0]):
            self._data = [Tensor().load_from(v) for v in data]
        elif is_float(data[0]):
            self._data = [float(v) for v in data]
        else:
            self._data = data
        self._assert_same_shape()

        return self

    def _assert_same_shape(self):
        if is_float(self._data[0]):
            return

        for i in range(self._len):
            if self._data[0]._len != self._data[i]._len:
                raise TypeError(f"Tensor has childs with different shapes")

    def __getitem__(self, index: int):
        return self._data[index]

    def __setitem__(self, index: int, value: Union[float, 'Tensor']):
        if is_tensor(self._data[0]):
            if not is_tensor(value):
                raise TypeError(f"Expected Tensor, received {type(value)}")
            if self._data[0].shape() != value.shape():
                raise TypeError(f"Expected shape{self._data[0].shape()} and received shape{value.shape()}")

            self._data[index] = value
        else:
            if not is_float(value):
                raise TypeError(f"Expected float, received {type(value)}")

            self._data[index] = float(value)

    def __mul__(self, other: float):
        if not is_float(other):
            raise TypeError(f"Expected float, received {type(other)}")

        res = [v * other for v in self._data]
        return Tensor().load_from(res)

    def __rmul__(self, other: float):
        return self.__mul__(other)

    def shape(self):
        if not isinstance(self._data[0], Tensor):
            return (self._len,)
        else:
            return (self._len,) + self._data[0].shape()


def is_float(val):
    return isinstance(val, float) or isinstance(val, int)

def is_list(val):
    return isinstance(val, list)

def is_tensor(val):
    return isinstance(val, Tensor)
