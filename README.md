# Rede Neural V2
Classe para uma rede neural feedforward com uma camada de entrada, camadas ocultas e uma camada de saída.

A rede é chamada `Neural`, uma `CNN` que você pode importar do package `Kinho`.

## Como instalar
```
pip install Kinho
```

## Métodos:
```py
def __init__(sizes=None, brain_path=None, eta=0.01, gpu=False):
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

#### Parâmetros:
- `sizes (list of floats)`: lista com o número de neurônios em cada camada da rede, onde o primeiro elemento da lista é a quantidade de neurônios na camada de entrada, o último é a quantidade de neurônios na camada de saída e os elementos intermediários são as quantidades de neurônios nas camadas ocultas.
- `brain_path(string)`: caminho de um arquivo `x.brain`, de um modelo pré-treinado que você já tenha salvo em seu diretório.
- `eta (float)`: taxa de aprendizado. Caso não seja definida, assumimos uma taxa padrão de __0.01__.
- `gpu (bool)`: se __True__, permite que a rede neural automaticamente mude o contexto para utilização da GPU para melhoria de performance.

Exemplo:
```py
from Kinho import Neural

net_without_imported_model = Neural(sizes=[10, 200, 300, 50, 5], eta=0.1, gpu=True)
'''
    Uma rede com 3 camadas ocultas (200, 300, 50). Uma camada de input com 10 entradas e,
    uma camada de output com 5 saídas. Taxa de aprendizado 0.1 e todos os pesos sinápticos
    aleatórios.
'''

net_with_imported_model = Neural(brain_path='./pre-trained/mnist_model.brain', eta=0.1, gpu=True)
'''
    Uma rede com a arquitetura e todos os pesos e biases importados de um modelo previamente treinado dentro do caminho <brain_path>.
'''

invalid_network = Neural(eta=0.1, gpu=True)
'''
    Um erro será gerado, pois é necessária a presença da arquitetura (sizes) ou modelo pré-treinado (brain_path).
'''
```

É __obrigatório__ passar o `sizes` ou o `brain_path`, caso contrário, um erro de tipo será gerado. Caso o usuário passe os dois, a rede irá priorizar o modelo importado, ou seja, o `brain_path`.

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

### export

#### Parâmetros:
- `filename (string)`: O nome do arquivo que você deseja dar para o novo arquivo de dados exportado.
- `path (string)`: caminho onde você deseja salvar seu arquivo. Coloque o diretório dentro da pasta de destino.

ex.: você poderá encontrar o arquivo neste diretório: `<path><filename>.brain`

#### Retorno:
Não há um retorno, mas você pode verificar se o arquivo se encontra no caminho especificado. Se sim, você já poderá compartilhar com outras aplicações e reutilizar os dados da sua rede e continuar o seu trabalho de onde parou.

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

> O tipo de dado `.brain` é um formato totalmente autoral deste projeto, suas especificações no momento não apresentam documentações, mas você pode conferir manualmente dentro da pasta `Kinho/brain`. Em breve, se necessário, haverá uma documentação mais explícita de como ler/criar este tipo de dado e quais especificações devem-se seguir para ser considerado um formato válido.
