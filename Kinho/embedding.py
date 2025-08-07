from .neural import Neural
import math
import numpy as np

class EncodingLayer:
    def __init__(self, size):
        self._size = size
        self._group = None
        self._dimension = None
        self._layer = None

        self._create_layer(size)

    def _create_layer(self, size):
        assert size > 0, 'Input dimension must be greater than zero'

        if size == 1:
            self._group = 'float'
            self._dimension = 1
        elif size <= 10:
            self._group = 'one-hot'
            self._dimension = size
        else:
            self._group = 'embedding'
            self._dimension = min(50, int(1.6 * math.log2(size)))

            limit = np.sqrt(1 / self._dimension)
            arr = np.random.uniform(-limit, limit, size=(size, self._dimension))
            self._layer = arr.tolist()

    @property
    def group(self):
        return self._group

    @property
    def size(self):
        return self._size

    @property
    def dimension(self):
        return self._dimension

    def __getitem__(self, key):
        if self._group == 'float':
            return [key]

        assert isinstance(key, int), f'Key must be an integer for one-hot/Embedding. Found {type(key)}'
        assert key >= 0 and key < self._size, f'Key must be >= 0 and < {self._size}. Key = {key}'

        if self._group == 'one-hot':
            ls = [0.0] * self._size
            ls[key] = 1.0
            return ls

        # Embedding
        return self._layer[key]


class EmbeddingNeural(Neural):
    def __init__(
        self,
        categories_dim=None,
        n_outputs=None,
        eta=0.01,
        mini_batch_size=1,
        multilabel=False
    ):
        self._encoding_layers = []
        self._history_table = []

        self._assert_in_dim(categories_dim)
        self._assert_out_dim(n_outputs)

        layer_0 = self._embedding(categories_dim)
        mid_layers = self._funelling(layer_0)
        sizes = [layer_0] + mid_layers + [n_outputs]

        super().__init__(
            sizes=sizes,
            brain_path=None,
            eta=eta,
            gpu=False,
            mini_batch_size=mini_batch_size,
            multilabel=multilabel
        )

    def _assert_in_dim(self, in_dim):
        if not in_dim:
            raise TypeError('You must set `in_dim`')

        if not isinstance(in_dim, list):
            raise TypeError('`in_dim` must be a list')

        for d in in_dim:
            if not isinstance(d, int):
                raise TypeError(f'`in_dim` must be a list of integers. Found a list of {type(d)}')

    def _assert_out_dim(self, out_dim):
        if not out_dim:
            raise TypeError('You must set `in_dim`')

        if not isinstance(out_dim, int):
            raise TypeError(f'`in_dim` must be a integer. Found {type(out_dim)}')

    def _embedding(self, in_dim):
        layer_len = 0
        for d in in_dim:
            el = EncodingLayer(d)
            self._encoding_layers.append(d)
            layer_len += el.dimension

        return layer_len

    def _funelling(self, neurons):
        layer_sz = int(2**(int(math.log2(neurons))))
        mid_layers = []

        while layer_sz >= 16:
            mid_layers.append(layer_sz)
            layer_sz //= 2

        return mid_layers

    def _convert_input(self, in_data):
        converted_data = []
        for i in range(len(in_data)):
            converted_data += self._encoding_layers[i][in_data[i]]
        return converted_data

    def _save_history(self, data):
        batch_calls = []
        pointer_start = 0
        for i in range(len(data)):
            el: EncodingLayer = self._encoding_layers[i]
            if el.group == 'embedding':
                batch_calls.append(
                    # (category, category value, start position)
                    (i, data[i], pointer_start)
                )
            pointer_start += el.dimension
        self._history_table.append(batch_calls)

    def __backPropagation(self, signals, targets):
        gradients = super().__backPropagation(signals, targets)

        for batch_id in len(gradients):
            gradient = gradients[batch_id]
            batch_calls = self._history_table[batch_id]

            for el_id, key, pointer_left in batch_calls:
                el: EncodingLayer = self._encoding_layers[el_id]
                vt = el[key]
                for i in range(el.dimension):
                    vt[i] -= gradient[pointer_left + i] * self.__eta

        self._history_table = []

    def send(self, input):
        converted_input = self._convert_input(input)
        return super().send(converted_input)

    def learn(self, input, output):
        converted_input = self._convert_input(input)
        self._save_history(input)
        return super().learn(converted_input, output)

    def cost(self, input, output):
        converted_input = self._convert_input(input)
        return super().cost(converted_input, output)

    def export(self, filename, path):
        raise NotImplementedError('export will be available soon...')

