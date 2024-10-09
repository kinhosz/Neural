from typing import List, Union
import warnings

class Tensor(object):
    """
    A class representing a multi-dimensional tensor.

    Args:
        shape: tuple[int]
            the tensor's size
    """

    _data: Union[List[float], List['Tensor']]

    def __init__(self, shape: tuple[int]):
        data = 0 if len(shape) == 1 else Tensor(shape[1:])
        self._data = [data] * shape[0]

    def __getitem__(self, index: int):
        return self._data[index]

    def __setitem__(self, index: int, value: Union[float, 'Tensor']):
        if type(self._data[0]) != type(value):
            warnings.warn(f"Expected data type: {type(self._data[0])}, but received: {type(value)}")
        if isinstance(value, Tensor) and self._data[0].shape() != value.shape():
            warnings.warn(f"Expected size: {self._data[0].shape()}, but received: {value.shape()}")

        self._data[index] = value

    def shape(self):
        if isinstance(self._data[0], int):
            return (len(self._data),)
        else:
            return (len(self._data),) + self._data[0].shape()
