# Neural Network [![Downloads](https://static.pepy.tech/badge/kinho)](https://pepy.tech/project/kinho) [![Downloads](https://static.pepy.tech/badge/kinho/month)](https://pepy.tech/project/kinho) [![Downloads](https://static.pepy.tech/badge/kinho/week)](https://pepy.tech/project/kinho)
Class for a feedforward neural network with an input layer, hidden layers, and an output layer.

The network is called `Neural`, a `CNN` that you can import from the `Kinho` package.

## How to install
`Python version: <3.12`
```
pip install Kinho
```

## Methods:
```py
def __init__(sizes=None, brain_path=None, eta=0.01, gpu=False, mini_batch_size=1, multilabel=False):
    pass

def send(input):
    pass

def learn(input, output):
    pass

def export(filename, path):
    pass

def cost(input, output):
    pass
```

### Neural (constructor)

#### Parameters:
- `sizes (list of floats)`: a list with the number of neurons in each layer of the network, where the first element in the list is the number of neurons in the input layer, the last is the number of neurons in the output layer, and the intermediate elements are the quantities of neurons in the hidden layers.
- `brain_path(string)`: the path to an x.brain file, a pre-trained model that you may have already saved in your directory.
- `eta (float)`: learning rate. If not defined, we assume a default rate of 0.01.
- `gpu (bool)`: if **True**, allows the neural network to automatically switch the context for GPU usage to improve performance.
- `mini_batch_size(int)`: power of 2. The network will process learning in parallel with the mini-batch size passed. If the GPU option is not selected, the network will continue to process data with the same mini-batch size but without parallelization. The size of the mini-batch depends on available memory, a size of 128 is usually better.
- `multilabel`: if for one instance, you need select more than one label.

Example:
```py
from Kinho import Neural

net_without_imported_model = Neural(sizes=[10, 200, 300, 50, 5], eta=0.1, gpu=True, mini_batch_size=16)
'''
    A network with 3 hidden layers (200, 300, 50). An input layer with 10 inputs and,
    an output layer with 5 outputs. Learning rate 0.1, and all synaptic weights
    randomized, with a mini-batch of size 16. 
'''

net_with_imported_model = Neural(brain_path='./pre-trained/mnist_model.brain', eta=0.1, gpu=True)
'''
    A network with the architecture and all weights and biases imported from a previously trained model inside the <brain_path>.
'''

invalid_network = Neural(eta=0.1, gpu=True)
'''
    An error will be generated because the presence of the architecture (sizes) or pre-trained model (brain_path) is required.
'''
```

It is __mandatory__ to pass sizes or brain_path; otherwise, a type error will be generated. If the user passes both, the network will prioritize the imported model, i.e., the `brain_path`.

### send

#### Parameters:
- `input (list of floats)`: The input values to be sent to the network.

#### Return:
`list[float]`: a list with the same size as the output layer of the network. For each position (label), there will be a float indicating the probability of the input corresponding to each label. The label with the highest probability should be considered as the prediction.

Example:
```py
input = [10, 2, 4, 4, 100, 90, 3, -1, 9, 10]
output = net.send(input)
print(output)
# [0.2, 0.05, 0.7, 0.05, 0.0]
'''
    The network assigns probabilities to each label. Therefore, there is a 20% chance of the label
    related to position 0 being the answer, and there is a 70% chance of the label related to
    position 2 being the answer to the input provided to the network. If multilabel flag was set to True,
    the network will return probabilities for each label independently.
'''
```

### learn

#### Parameters:
- `input (list of floats)`: The input values to be sent to the network.
- `output (list of floats)`: The expected probability of each label being related to the input. Note that the sum of all probabilities should be equal to __1.0.__, ignore the sum if `multilabel` flag was set to __True__.

#### Return:
There is no return; the network only learns using __backpropagation__ and updates its weights and biases.

Example:
```py
input = [10, 2, 4, 4, 100, 90, 3, -1, 9, 10]
output = [0.0, 0.0, 1.0, 0.0, 0.0]

net.learn(input, output)
'''
    The response for the given input should be 2. Therefore, the network receives
    the expected output from the sea of probabilities and learns to reduce the error.
'''
```

### export

#### Parameters:
- `filename (string)`: The name you want to give to the new exported data file.
- `path (string)`: The path where you want to save your file. Place the directory inside the destination folder.

e.g., you can find the file in this directory: `<path><filename>.brain`

#### Return:
There is no return, but you can check if the file exists at the specified path. If it does, you can already share it with other applications and reuse the network's data and continue your work from where you left off.

### cost

#### Parameters:
- `input (list of floats)`:  The input values to be sent to the network.
- `output (list of floats)`:  The expected probability of each label being related to the input. Note that the sum of all probabilities should be equal to __1.0__.

#### Return:
`float`: the value of the mean of the squares of the differences between the output expected by the user and the output generated by the network.

Example:
```py
input = [10, 2, 4, 4, 100, 90, 3, -1, 9, 10]
output = [0.0, 0.0, 1.0, 0.0, 0.0]

mse = net.cost(input, output)
print(mse)
# 0.027
```

> The `.brain` data type is a completely proprietary format of this project; its specifications currently do not have documentation, but you can check manually inside the `Kinho/brain` folder. Soon, if necessary, there will be more explicit documentation on how to read/create this data type and what specifications must be followed to be considered a valid format.
