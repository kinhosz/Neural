import numpy as np
from numpy.typing import NDArray

from .lib import ConvLayer, MaxPooling
from .tensor import Tensor
from .neural import Neural

from .helper import calculate_pooling_layers

class ConvNeural(Neural):
    """
    Class representing a convolutional neural network.

    Args:
        in_shape (tuple[int, int, int]): Dimensions of the network input (channels, width, height).
        filters (list[int]): List containing the number of filters for each convolutional layer.
                                The length should be equals to `calculate_pooling_layers`
        fc_layers (list[int]): List containing the number of neurons in each fully connected layer,
                                including the output layer.
    """
    def __init__(
        self,
        in_shape: tuple[int, int, int],
        filters: list[int],
        fc_layers: list[int]
    ) -> None:

        self._assert_args(in_shape, filters, fc_layers)

        super().__init__(sizes=[filters[-1]] + fc_layers)

        self._backbone = []
        res_weights, res_biases = self._build_backbone_tensors([in_shape[0]] + filters)

        self._build_architecture(res_weights, res_biases)

    def _assert_args(self, input_shape, filters, fc_layers):
        assert isinstance(input_shape, tuple), "The `input_shape` must be a tuple."
        assert len(input_shape) == 3, "The tuple must contain exactly 3 elements."
        for el in input_shape:
            assert isinstance(el, int), "The values of `input_shape` must be an integer"
            assert el > 0, "The values of `input_shape` must be greater than zero"

        assert isinstance(filters, list), "The `filters` must be a list"
        assert len(filters) == calculate_pooling_layers(input_shape[1], input_shape[2]), \
            "`filters` length should be equals to `calculate_pooling_layers"
        for el in filters:
            assert isinstance(el, int), "The values of `filters` must be an integer"
            assert el > 0, "The values of `filters` must be greater than zero"

        assert isinstance(fc_layers, list), "The `fc_layers` must be a list"
        assert len(fc_layers) > 0, "`fc_layers` length must be greater than zero"
        for el in fc_layers:
            assert isinstance(el, int), "The values of `fc_layers` must be an integer"
            assert el > 0, "The values of `fc_layers` must be greater than zero"

    def _build_backbone_tensors(self, filters):
        resources_weights = []
        resources_biases = []

        for i in range(1, len(filters)):
            resources_weights.append(
                Tensor((filters[i], filters[i-1], 3, 3))
            )
            resources_biases.append(
                Tensor((filters[i],))
            )

        return resources_weights, resources_biases

    def _build_architecture(self, res_weights, res_biases):
        for res_weight, res_biase in zip(res_weights, res_biases):
            self._backbone.append(ConvLayer(weight=res_weight, biase=res_biase))
            self._backbone.append(MaxPooling())

    def _pre_nn_feed_forward(self, data: Tensor) -> NDArray:
        for func in self._backbone:
            data = func.send(in_data=data) # (K, 1, 1)

        def k11_to_11k(tensor):
            return np.array([[[tensor[i][0][0] for i in range(tensor.shape()[0])]]])

        return k11_to_11k(data) # (1, 1, Z)

    def _feed_forward(self, data: Tensor) -> Tensor:
        nn_in_data = self._pre_nn_feed_forward(data)
        nn_out_data = super()._feed_forward(nn_in_data)

        return Tensor().load_from(nn_out_data.tolist())

    def _out_to_target(self, tensor: Tensor) -> NDArray:
        Z = tensor.shape()[2]
        return np.array([[[tensor[0][0][z] for z in range(Z)]]])  # (1, 1, Z)

    def _backpropagation(self, in_data: Tensor, out_data: Tensor) -> None:
        # Calling `_pre_nn_feed_forward` first because we must cache `in_data` for backbone's layer
        nn_in_data = self._pre_nn_feed_forward(in_data)
        nn_out_data = self._out_to_target(out_data)

        nn_gradients = super()._backpropagation(signals=nn_in_data, targets=nn_out_data)

        K = nn_gradients.shape[2]
        gradients = Tensor((K, 1, 1))
        for k in range(K):
            gradients[k][0][0] = nn_gradients[0][0][k]

        for func in reversed(self._backbone):
            gradients = func.learn(gradients=gradients)

    def _cost(self, in_data: Tensor, out_data: Tensor) -> NDArray:
        nn_in_data = self._pre_nn_feed_forward(in_data)
        nn_out_data = self._out_to_target(out_data)

        predict = super()._feed_forward(nn_in_data)  # (1, 1, Z), raw pre-MSE

        return self._layer[-1].send(predict, nn_out_data)

    def send(self, in_data: Tensor) -> Tensor:
        """
        Processes the input tensor, which represents an image with 3 domains, 
        and returns a tensor with a single domain.

        Args:
            in_data (Tensor): A 3-dimensional tensor (channels, width, height) representing an image.
                            The tensor must follow the format where the first domain is the
                            color channels and the remaining two are the image's spatial dimensions.

        Returns:
            Tensor: A 1-dimensional tensor that results from the processing of the input, 
                    representing a feature vector.
        """
        if len(in_data.shape()) != 3:
            raise TypeError("in_data must be a 3-Dimensional")

        out_data = self._feed_forward(in_data)
        return out_data[0][0]

    def learn(self, in_data: Tensor, out_data: Tensor) -> None:
        """
        Trains the network using the provided input and output data.

        Args:
            in_data (Tensor): A 3-dimensional tensor representing the input data (e.g., an image).
            out_data (Tensor): A 1-dimensional tensor representing the expected output values for
                                the corresponding features.

        The network will adjust its internal parameters based on the input data and the expected output features.
        """
        if len(in_data.shape()) != 3:
            raise TypeError("in_data must be a 3-Dimensional")
        if len(out_data.shape()) != 1:
            raise TypeError("out_data must be a 1-Dimensional")

        out_tensor = Tensor((1, 1) + out_data.shape())
        out_tensor[0][0] = out_data

        self._backpropagation(in_data, out_tensor)

    def cost(self, in_data: Tensor, out_data: Tensor) -> float:
        """
        Computes the Mean Squared Error (MSE) between the predicted output and the expected output.

        Args:
            in_data (Tensor): A 3-dimensional tensor representing the input data.
            out_data (Tensor): A 1-dimensional tensor representing the expected output data.

        Returns:
            float: The Mean Squared Error (MSE) between the predicted and expected output.
        """
        if len(in_data.shape()) != 3:
            raise TypeError("in_data must be a 3-Dimensional")
        if len(out_data.shape()) != 1:
            raise TypeError("out_data must be a 1-Dimensional")

        out_tensor = Tensor((1, 1) + out_data.shape())
        out_tensor[0][0] = out_data

        cost_error = self._cost(in_data, out_tensor)
        return cost_error[0]

    def export(self):
        # TODO: export the dense head via `super().export(...)` (already
        # compatible, same `self._layer`) and the backbone (`self._backbone`)
        # into a separate file, since its weight format may still change.
        pass
