import numpy as np

class Tensor(object):
    """
    A class representing a multi-dimensional tensor.

    Internally backed by a numpy ndarray for speed (CPU-only for now), but
    the public surface (shape/indexing/load_from/mul) stays the same on
    purpose: callers should keep treating this as an opaque, hardware-
    agnostic input/output type, never touch `_array` directly.

    Args:
        shape: tuple[int] = None
            the tensor's size

    """

    def __init__(self, shape: tuple[int] = None, uniform=False, scale=1.0):
        self._array = None

        if shape:
            if uniform:
                self._array = np.random.uniform(low=-scale, high=scale, size=shape)
            else:
                self._array = np.zeros(shape, dtype=float)

    @staticmethod
    def from_ndarray(array: np.ndarray) -> 'Tensor':
        """Wraps an existing ndarray (no copy if it's already float64) -
        the fast path for the numpy<->Tensor boundary in lib/cpu/*."""
        t = Tensor()
        t._array = np.asarray(array, dtype=float)
        return t

    def load_from(self, data) -> 'Tensor':
        if is_tensor(data):
            self._array = np.array(data._array, dtype=float)
        else:
            self._array = np.array(data, dtype=float)

        if self._array.ndim == 0 or self._array.size == 0:
            raise TypeError(f"Expected list with len > 0, received {data}")

        return self

    @staticmethod
    def from_trusted_list(data: list) -> 'Tensor':
        """Kept for API compatibility with earlier callers - numpy's own
        array construction is already the fast path, so this is now just
        an alias for load_from/from_ndarray."""
        return Tensor.from_ndarray(np.array(data, dtype=float))

    def __getitem__(self, index):
        sub = self._array[index]
        if np.ndim(sub) == 0:
            return float(sub)
        return Tensor.from_ndarray(sub)

    def __setitem__(self, index, value):
        if self._array.ndim == 1:
            if not is_float(value):
                raise TypeError(f"Expected float, received {type(value)}")
            self._array[index] = float(value)
        else:
            if not is_tensor(value):
                raise TypeError(f"Expected Tensor, received {type(value)}")
            if self._array[index].shape != value.shape():
                raise TypeError(f"Expected shape{self._array[index].shape} and received shape{value.shape()}")
            self._array[index] = value._array

    def __mul__(self, other: float):
        if not is_float(other):
            raise TypeError(f"Expected float, received {type(other)}")
        return Tensor.from_ndarray(self._array * other)

    def __rmul__(self, other: float):
        return self.__mul__(other)

    def to_list(self):
        return self._array.tolist()

    def to_ndarray(self) -> np.ndarray:
        """Returns the underlying ndarray directly (a view, not a copy) -
        the fast path for the numpy<->Tensor boundary in lib/cpu/*."""
        return self._array

    def shape(self):
        return self._array.shape


def is_float(val):
    return isinstance(val, float) or isinstance(val, int)

def is_list(val):
    return isinstance(val, list)

def is_tensor(val):
    return isinstance(val, Tensor)
