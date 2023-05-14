# Rede Neural (GPU/CPU Mode)
Classe para uma rede neural feedforward com uma camada de entrada, camadas ocultas e uma camada de saída.

A rede é chamada `Neural`, uma `CNN` que você pode importar do package `Deep`.

## Métodos:
```py
def __init__(sizes, eta=0.01, random_weights=True, gpu=False):
    pass

def send(input):
    pass

def learn(input, output):
    pass

def cost(input, output):
    pass
```

### Neural (constructor)

#### Parâmetros:
- `sizes (list of floats)`: lista com o número de neurônios em cada camada da rede, onde o primeiro elemento da lista é a quantidade de neurônios na camada de entrada, o último é a quantidade de neurônios na camada de saída e os elementos intermediários são as quantidades de neurônios nas camadas ocultas.
- `eta (float)`: taxa de aprendizado. Caso não seja definida, assumimos uma taxa padrão de __0.01__.
- `random_weights (bool)`: se __True__, os pesos sinápticos serão inicializados aleatoriamente. Caso contrário, será criada uma rede neural com todos os pesos sinápticos iguais a zero.
- `gpu (bool)`: se __True__, permite que a rede neural automaticamente mude o contexto para utilização da GPU para melhoria de performance.

Exemplo:
```py
from Deep import Neural

net = Neural([10, 200, 300, 50, 5], eta=0.1, random_weights=True, gpu=True)
'''
    Uma rede com 3 camadas ocultas (200, 300, 50). Uma camada de input com 10 entradas e,
    uma camada de output com 5 saídas. Taxa de aprendizado 0.1 e todos os pesos sinápticos
    aleatórios.
'''
```

### send

#### Parâmetros:
- `input (list of floats)`: Os valores de entrada que serão enviados para a rede.

#### Retorno:
`list[float]`: uma lista com o mesmo tamanho da camada de saída da rede. Para posição(rótulo), existirá um float informando a probabilidade da entrada corresponder com cada rótulo. Deve-se considerar como predição o rótulo no qual tiver maior probabilidade.

Exemplo:
```py
input = [10, 2, 4, 4, 100, 90, 3, -1, 9, 10]
output = net.send(input)
print(output)
# [0.2, 0.05, 0.7, 0.05, 0.0]
'''
    A rede atribuiu a probabilidade para cada rótulo. Logo, há 20% de chance do rótulo
    relacionado a posição 0 ser a resposta, e há 70% de chance do rótulo relacionado a
    posição 2 ser a a resposta da entrada cedida à rede.
'''
```

### learn

#### Parâmetros:
- `input (list of floats)`: Os valores de entrada que serão enviados para a rede.
- `output (list of floats)`: A probabilidade esperada de cada rótulo estar relacionado com o input. Note que a soma de todas as probabilidades deverá ser igual a __1.0__.

#### Retorno:
Não há retorno, a rede apenas aprende utilizando __backpropagation__ e atualiza seus pesos e biases.

Exemplo:
```py
input = [10, 2, 4, 4, 100, 90, 3, -1, 9, 10]
output = [0.0, 0.0, 1.0, 0.0, 0.0]

net.learn(input, output)
'''
    A resposta para o input dado deve ser 2. Logo, a rede recebe
    o output esperado do mar de probabilidades e aprende a diminuir o erro.
'''
```

### cost

#### Parâmetros:
- `input (list of floats)`: Os valores de entrada que serão enviados para a rede.
- `output (list of floats)`: A probabilidade esperada de cada rótulo estar relacionado com o input. Note que a soma de todas as probabilidades deverá ser igual a __1.0__.

#### Retorno:
`float`: o valor da média do quadrado das diferenças entre o output esperado pelo usuário e o output gerado pela rede.

Exemplo:
```py
input = [10, 2, 4, 4, 100, 90, 3, -1, 9, 10]
output = [0.0, 0.0, 1.0, 0.0, 0.0]

mse = net.cost(input, output)
print(mse)
# 0.027
```