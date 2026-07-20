import csv
from Kinho import EmbeddingNeural
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

dimensions = []

def csv_reader(filename):
  with open(filename, mode='r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    dataset = []
    for line in reader:
      data = {}
      for k in line:
        data[k] = line[k]
      dataset.append(data)

    return dataset

def pre_convert_geral(dataset, table, label, cnae=False):
  total_labels = len(table.keys()) + 1 # For non-present data
  dimension = total_labels if total_labels > 2 else 1
  dimensions.append(dimension)

  for data in dataset:
    k = data[label]
    if cnae:
      k = data[label][:5]

    label_id = table.get(k, total_labels-1)
    data['labels'].append(label_id)

def pre_convert_municipio(dataset):
  cities = {
    'FORTALEZA': 0,
    'RECIFE': 1,
    'JOAO PESSOA': 2,
    'MACEIO': 3,
    'JABOATAO DOS GUARARAPES': 4,
    'OLINDA': 5,
    'PAULISTA': 6,
    'CARUARU': 7,
    'CAUCAIA': 8,
    'IPOJUCA': 9,
    'CAMARAGIBE': 10,
    'CABO DE SANTO AGOSTINHO': 11,
    'MARACANAU': 12,
    'CABEDELO': 13,
    'GRAVATA': 14,
    'SANTA RITA': 15,
    'SAO LOURENCO DA MATA': 16,
    'VITORIA DE SANTO ANTAO': 17,
    'AQUIRAZ': 18,
    'ABREU E LIMA': 19,
    'GARANHUNS': 20,
    'EUSEBIO': 21,
    'IGARASSU': 22,
    'CARPINA': 23,
    'HORIZONTE': 24,
    'BAYEUX': 25,
    'RIO LARGO': 26,
    'PACAJUS': 27,
    'ARACATI': 28,
    'CASCAVEL': 29,
    'ITAITINGA': 30,
    'CONDE': 31,
    'TAMANDARE': 32,
    'MARANGUAPE': 33,
    'MORENO': 34,
    'BATURITE': 35,
    'SAO JOSE DA COROA GRANDE': 36,
    'PARACURU': 37,
    'PACATUBA': 38,
    'SAO GONCALO DO AMARANTE': 39,
    'BEZERROS': 40,
    'BARREIROS': 41,
    'PINDORETAMA': 42,
    'ILHA DE ITAMARACA': 43,
    'BEBERIBE': 44,
    'PARAIPABA': 45,
    'LUCENA': 46,
    'PAUDALHO': 47,
    'POMBOS': 48,
    'RIO FORMOSO': 49,
    'REDENCAO': 50,
    'SIRINHAEM': 51,
    'GUARAMIRANGA': 52,
    'ITAPISSUMA': 53,
    'SAO CAITANO': 54,
  }

  pre_convert_geral(dataset, cities, 'municipio')

def pre_convert_segmento_cliente(dataset):
  segmento_cliente = {
    'Cafeteria': 0,
    'Atacarejo & Supermercado': 1,
    'Lanches & Salgados': 2,
    'Conveniência': 3,
    'Japonês & Chinês & Asiático': 4,
    'Office - Escolas & Escritórios & Consultórios & Associações': 5,
    'Padaria': 6,
    'Churrascaria': 7,
    'Bar': 8,
    'Clube': 9,
    'Distribuidor': 10,
    'Hospital': 11,
    'Hotel': 12,
    'Pizza & Massas': 13,
    'Doces & Confeitaria': 14,
    'Saudável': 15,
    'Açaí & Sorvetes': 16,
    'Restaurante Brasileiro': 17,
    'Cozinha Industrial': 18,
    'Mercearia & Mercado': 19,
    'Hambúrguer': 20
  }

  pre_convert_geral(dataset, segmento_cliente, 'segmento_cliente')

def pre_convert_natureza_juridica(dataset):
  natureza = {
    '213-5 - Empresário (Individual)': 0,
  }
  
  pre_convert_geral(dataset, natureza, 'natureza_juridica')

def pre_convert_fonte_cliente(dataset):
  fonte_cliente = {
    'Fonte 1': 0,
    'Fonte 2': 1,
    'Fonte 3': 2,
    'Fonte 4': 3,
    'Fonte 5': 4
  }

  pre_convert_geral(dataset, fonte_cliente, 'fonte_cliente')

def pre_convert_cnae(dataset):
  cnae_codigo = {
    '56.11': 0,
    '56.20': 1,
    '47.12': 2,
    '10.91': 3,
    '47.21': 4,
    '47.23': 5,
    '56.12': 6,
    '47.29': 7,
    '55.10': 8,
    '47.24': 9,
  }

  pre_convert_geral(dataset, cnae_codigo, 'cnae_codigo', True)

def pre_convert_capital_social(dataset):
  capital_social = {
    '(3000, 5000]': 0,
    '(200, 1000]': 1,
    '(8000, 10000]': 2,
    '(30000, 50000]': 3,
    '(15000, 20000]': 4,
    '(1500, 2000]': 5,
    '(20000, 30000]': 6,
    '(2000, 3000]': 7,
    '(70000, 100000]': 8,
    '(0, 1]': 9,
    '(10000, 15000]': 10,
    '(5000, 8000]': 11,
    '(100000, 200000]': 12,
    '(1, 200]': 13,
    '(1000, 1500]': 14,
    '(50000, 70000]': 15,
    '(200000, 1000000]': 16,
    '(1000000, 10000000]': 17,
    '(10000000, 100000000000]': 18,
  }

  pre_convert_geral(dataset, capital_social, 'capital_social')

def pre_convert_idade_cnpj(dataset):
  idade_cnpj = {
    '(900, 1150]': 0,
    '(1600, 2000]': 1,
    '(150, 250]': 2,
    '(3000, 4000]': 3,
    '(450, 600]': 4,
    '(50, 100]': 5,
    '(0, 25]': 6,
    '(1300, 1600]': 7,
    '(600, 750]': 8,
    '(250, 350]': 9,
    '(4000, 5000]': 10,
    '(750, 900]': 11,
    '(2000, 2400]': 12,
    '(2400, 3000]': 13,
    '(350, 450]': 14,
    '(100, 150]': 15,
    '(25, 50]': 16,
    '(5000, 8000]': 17,
    '(1150, 1300]': 18,
    '(8000, 25000]': 19,
  }

  pre_convert_geral(dataset, idade_cnpj, 'idade_cnpj')

def pre_convert_serasa_contagem_negativacoes(dataset):
  table = {
    '0': 0,
  }
  
  pre_convert_geral(dataset, table, 'serasa_contagem_negativacoes')

def pre_convert_serasa_contagem_protestos(dataset):
  table = {
    '0': 0,
  }

  pre_convert_geral(dataset, table, 'serasa_contagem_protestos')

def pre_convert_serasa_divida_mais_recente(dataset):
  for data in dataset:
    if data['serasa_idade_divida_mais_recente'] == '':
      data['serasa_idade_divida_mais_recente'] = 'NONE'

  table = {
    'NONE': 0,
  }

  pre_convert_geral(dataset, table, 'serasa_idade_divida_mais_recente')

def pre_convert_serasa_socio_tem_negativacao(dataset):
  table = {
    '0': 0,
  }

  pre_convert_geral(dataset, table, 'serasa_socio_tem_negativacao')

def pre_convert_inadimplente(dataset):
  for data in dataset:
    data['inadimplente'] = int(data['inadimplente'])

def pre_convert(dataset, ignore_inadimplente=False):
  global dimensions
  dimensions = []
  for data in dataset:
    data['labels'] = []

  pre_convert_municipio(dataset)
  pre_convert_segmento_cliente(dataset)
  pre_convert_natureza_juridica(dataset)
  pre_convert_fonte_cliente(dataset)
  pre_convert_cnae(dataset)
  pre_convert_capital_social(dataset)
  pre_convert_idade_cnpj(dataset)
  pre_convert_serasa_contagem_negativacoes(dataset)
  pre_convert_serasa_contagem_protestos(dataset)
#   pre_convert_serasa_divida_mais_recente(dataset)
#   pre_convert_serasa_socio_tem_negativacao(dataset)
  if not ignore_inadimplente:
    pre_convert_inadimplente(dataset)
  print()

def print_info(y_score, y_true, filename):
  txt = ""
  for y, yf in zip(y_score, y_true):
    txt += f"val:{round(y, 2)} target: {round(yf, 2)} dif: {round(abs(yf-y), 2)}\n"
  f = open('log.txt', 'w')
  f.write(txt)
  f.close() 

  # Calcula os pontos da curva ROC
  fpr, tpr, thresholds = roc_curve(y_true, y_score)

  # Calcula a área sob a curva (AUC)
  roc_auc = auc(fpr, tpr)

  # Plota a curva
  plt.figure()
  plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # linha diagonal (modelo aleatório)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Curva ROC')
  plt.legend(loc='lower right')
  plt.grid(True)
  plt.savefig(filename)
  plt.close()

  return roc_auc

def print_csv(prediction, resource, filename):
  txt = "id,target\n"
  for p, id in zip(prediction, resource):
    txt += f"{id},{p}\n"

  f = open(filename, "w")
  f.write(txt)
  f.close()

#########
class Model(EmbeddingNeural):
  def predict(self, in_data):
    ret = []
    for d in in_data:
      ret.append(
        self.send(d)[0]
      )

    return ret

  def fit(self, in_data, target):
    len_x_train = int(len(in_data) * 0.9)
    x_train = in_data[:len_x_train]
    y_train = target[:len_x_train]

    x_validation = in_data[len_x_train:]
    y_validation = target[len_x_train:]

    EPOCH = 100
    CHECKER_EPOCH = int(EPOCH * 0.1)
    better_auc = 0.0
    for i in range(EPOCH):
      for j in range(len(x_train)):
        d = x_train[j]
        t = y_train[j]
        self.learn(d, [t])

        perc = round(100.0 * (i * len_x_train + j) / (EPOCH * len_x_train), 2)
        print(f"\rLoading: {perc}%  -- auc = {better_auc}", end='', flush=True)

      if i % CHECKER_EPOCH == 0:
        pred = self.predict(x_validation)
        auc = print_info(pred, y_validation, 'wip_train.png')
        better_auc = max(better_auc, round(auc, 4))

    print(f"\r                                               ", end='', flush=True)
#########

def main():
  random.seed(42)

  dataset = csv_reader('train.csv')
  pre_convert(dataset)

  random.shuffle(dataset)

  dataset_len = len(dataset)
  train_len = int(dataset_len * 0.9)

  train = dataset[:train_len]
  test = dataset[train_len:]

  x_train = [t['labels'] for t in train]
  y_train = [t['inadimplente'] for t in train]

  x_test = [t['labels'] for t in test]
  y_test = [t['inadimplente'] for t in test]


  model = Model(
    categories_dim=dimensions,
    n_outputs=1,
    eta=0.01,
    mini_batch_size=64,
  )

  model.fit(x_train, y_train)
  pred = model.predict(x_test)
  auc = print_info(pred, y_test, f'kaggle_model_validation.png')
  print(f"\nModel Test:")
  print("  AUC :", auc)

  #model.export('praso_kaggle', '')

  ## TEST FINAL
  dataset = csv_reader('test.csv')
  pre_convert(dataset, True)

  x_test = [t['labels'] for t in dataset]
  z_test = [t['id'] for t in dataset]

  pred = model.predict(x_test)
  print(f"\nModel Validation:")
  print("  AUC :", auc)

  print_csv(pred, z_test, 'sample_submission.csv')


if __name__ == "__main__":
  main()