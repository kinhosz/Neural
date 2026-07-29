from numpy.typing import NDArray

from .lib import ConvLayer, MaxPooling, LeakyReLU, TensorSigmoid2, Sigmoid2, LeakyReLUDense
from .tensor import Tensor
from .neural import Neural

from .helper import calculate_pooling_layers

KERNEL_SIZE = 3  # ConvLayer only supports 3x3 kernels today (see lib/cpu/convolution.py)

BACKBONE_ACTIVATIONS = {
    None: None,
    "sigmoid2": TensorSigmoid2,
    "leaky_relu": LeakyReLU,
}

DENSE_ACTIVATIONS = {
    "sigmoid2": Sigmoid2,
    "leaky_relu": LeakyReLUDense,
}

class ConvNeural(Neural):
    """
    Class representing a convolutional neural network.

    Args:
        in_shape (tuple[int, int, int]): Dimensions of the network input (channels, width, height).
        filters (list[int]): List containing the number of filters for each convolutional layer.
                                The length should be equals to `calculate_pooling_layers`
        fc_layers (list[int]): List containing the number of neurons in each fully connected layer,
                                including the output layer.
        batch_size (int): Size `learn()` requires for every incoming batch (must be a power of 2,
                                same constraint as `Neural`'s `mini_batch_size`). `send()`/`cost()`
                                stay flexible and also accept a single, unbatched image.
        eta (float): Learning rate, forwarded to the dense head (`Neural`).
        activation (str | None): Activation applied after each ConvLayer, before MaxPooling.
                                One of "sigmoid2", "leaky_relu" (default), or None (no
                                activation - a stack of purely linear conv+pool stages, which
                                doesn't learn: see the "leaky_relu" experiment in git history).
        dense_activation (str): Hidden activation for the dense head (`fc_layers`). One of
                                "sigmoid2" or "leaky_relu" (default - required to avoid
                                saturating on the leaky_relu backbone's unbounded output).
                                The output activation (Softmax/Sigmoid2) is unaffected.
    """
    def __init__(
        self,
        in_shape: tuple[int, int, int],
        filters: list[int],
        fc_layers: list[int],
        batch_size: int = 1,
        eta: float = 0.01,
        activation: str = "leaky_relu",
        dense_activation: str = "leaky_relu"
    ) -> None:

        self._assert_args(in_shape, filters, fc_layers)
        assert activation in BACKBONE_ACTIVATIONS, \
            "`activation` must be one of {}".format(list(BACKBONE_ACTIVATIONS.keys()))
        assert dense_activation in DENSE_ACTIVATIONS, \
            "`dense_activation` must be one of {}".format(list(DENSE_ACTIVATIONS.keys()))

        self._batch_size = batch_size

        super().__init__(
            sizes=[filters[-1]] + fc_layers,
            eta=eta,
            hidden_activation=DENSE_ACTIVATIONS[dense_activation],
            mini_batch_size=batch_size
        )

        self._backbone = []
        res_weights, res_biases = self._build_backbone_tensors([in_shape[0]] + filters)

        self._build_architecture(res_weights, res_biases, eta, BACKBONE_ACTIVATIONS[activation])

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
            # Var(sum of fan_in weight*input terms) = fan_in * Var(weight);
            # solving Var(weight) = limit^2/3 for Var(sum) == 1 gives this.
            fan_in = filters[i-1] * KERNEL_SIZE * KERNEL_SIZE
            limit = (3 / fan_in) ** 0.5

            resources_weights.append(
                Tensor((filters[i], filters[i-1], KERNEL_SIZE, KERNEL_SIZE), uniform=True, scale=limit)
            )
            resources_biases.append(
                Tensor((filters[i],))
            )

        return resources_weights, resources_biases

    def _build_architecture(self, res_weights, res_biases, eta, activation_cls):
        for res_weight, res_biase in zip(res_weights, res_biases):
            self._backbone.append(ConvLayer(weight=res_weight, biase=res_biase, eta=eta))
            if activation_cls is not None:
                self._backbone.append(activation_cls())
            self._backbone.append(MaxPooling())

    def _pre_nn_feed_forward(self, data: Tensor) -> NDArray:
        for func in self._backbone:
            data = func.send(in_data=data) # (BATCH, K, 1, 1)

        arr = data.to_ndarray()
        batch, k = arr.shape[0], arr.shape[1]

        return arr.reshape(batch, 1, k) # (BATCH, 1, Z)

    def _feed_forward(self, data: Tensor) -> Tensor:
        nn_in_data = self._pre_nn_feed_forward(data)
        nn_out_data = super()._feed_forward(nn_in_data)

        return Tensor().load_from(nn_out_data)

    def _out_to_target(self, tensor: Tensor) -> NDArray:
        return tensor.to_ndarray() # (BATCH, 1, Z)

    def _backpropagation(self, in_data: Tensor, out_data: Tensor) -> None:
        # Calling `_pre_nn_feed_forward` first because we must cache `in_data` for backbone's layer
        nn_in_data = self._pre_nn_feed_forward(in_data)
        nn_out_data = self._out_to_target(out_data)

        nn_gradients = super()._backpropagation(signals=nn_in_data, targets=nn_out_data)

        batch, k = nn_gradients.shape[0], nn_gradients.shape[2]
        gradients = Tensor().load_from(nn_gradients.reshape(batch, k, 1, 1))

        for func in reversed(self._backbone):
            gradients = func.learn(gradients=gradients)

    def _cost(self, in_data: Tensor, out_data: Tensor) -> NDArray:
        nn_in_data = self._pre_nn_feed_forward(in_data)
        nn_out_data = self._out_to_target(out_data)

        predict = super()._feed_forward(nn_in_data)  # (1, 1, Z), raw pre-MSE

        return self._layer[-1].send(predict, nn_out_data)

    def _as_batch(self, in_data: Tensor) -> Tensor:
        """Wraps a single image (K, H, W) into a batch of 1 (1, K, H, W)."""
        return Tensor().load_from(in_data.to_ndarray()[None])

    def send(self, in_data: Tensor) -> Tensor:
        """
        Processes the input tensor, which represents one or more images with 3 domains,
        and returns a tensor with a single domain per image.

        Args:
            in_data (Tensor): Either a 3-dimensional tensor (channels, width, height)
                            representing a single image, or a 4-dimensional tensor
                            (batch, channels, width, height) representing a batch of
                            images. The channel domain comes first, the remaining two
                            (or three) domains are the batch/spatial dimensions.

        Returns:
            Tensor: A 1-dimensional feature vector (single image in) or a 2-dimensional
                    (batch, features) tensor (batch in).
        """
        ndim = len(in_data.shape())
        if ndim not in (3, 4):
            raise TypeError("in_data must be a 3-Dimensional image or a 4-Dimensional batch")

        single = ndim == 3
        batch_data = self._as_batch(in_data) if single else in_data

        out_data = self._feed_forward(batch_data)  # (BATCH, 1, Z)
        batch, z = out_data.shape()[0], out_data.shape()[2]
        out_arr = out_data.to_ndarray().reshape(batch, z)

        return Tensor().load_from(out_arr[0]) if single else Tensor().load_from(out_arr)

    def learn(self, in_data: Tensor, out_data: Tensor) -> None:
        """
        Trains the network using the provided input and output data.

        Args:
            in_data (Tensor): A 4-dimensional tensor [BATCH][K][H][W] with exactly the
                                `batch_size` this network was created with.
            out_data (Tensor): A 2-dimensional tensor [BATCH][Z] with the expected output
                                values for each sample in the batch.

        The network will adjust its internal parameters based on the input data and the expected output features.
        """
        if len(in_data.shape()) != 4:
            raise TypeError("in_data must be a 4-Dimensional batch [BATCH][K][H][W]")
        if len(out_data.shape()) != 2:
            raise TypeError("out_data must be a 2-Dimensional batch [BATCH][Z]")
        if in_data.shape()[0] != self._batch_size or out_data.shape()[0] != self._batch_size:
            raise ValueError(
                "in_data/out_data batch size must match the configured batch_size={}".format(self._batch_size)
            )

        batch, z = out_data.shape()
        out_tensor = Tensor().load_from(out_data.to_ndarray().reshape(batch, 1, z))

        self._backpropagation(in_data, out_tensor)

    def cost(self, in_data: Tensor, out_data: Tensor) -> float:
        """
        Computes the Mean Squared Error (MSE) between the predicted output and the expected output.

        Args:
            in_data (Tensor): A 3-dimensional (single image) or 4-dimensional (batch) tensor,
                                matching `out_data`'s single/batch shape.
            out_data (Tensor): A 1-dimensional (single image) or 2-dimensional (batch) tensor
                                with the expected output data.

        Returns:
            float: The Mean Squared Error (MSE) between the predicted and expected output.
        """
        in_single = len(in_data.shape()) == 3
        out_single = len(out_data.shape()) == 1
        if in_single != out_single:
            raise TypeError("in_data and out_data must both be single-sample or both be batched")

        in_batch = self._as_batch(in_data) if in_single else in_data
        out_batch = Tensor().load_from(out_data.to_ndarray()[None]) if out_single else out_data

        batch, z = out_batch.shape()
        out_tensor = Tensor().load_from(out_batch.to_ndarray().reshape(batch, 1, z))

        cost_error = self._cost(in_batch, out_tensor)
        return cost_error[0]

    def export(self):
        # TODO: export the dense head via `super().export(...)` (already
        # compatible, same `self._layer`) and the backbone (`self._backbone`)
        # into a separate file, since its weight format may still change.
        pass
