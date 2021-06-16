# DataScience Suite
Este pacote tem o objetivo de agregar os diversos códigos autorais que criamos para todo o pipeline de produção dos modelos.
Atualmente é apenas um teste e possui apenas a library dssuite(ring.py) com algumas funções que facilitam leitura, tuning e avaliação dos modelos.

## Documentação rudimentar
### SampleStress(Classe)
 Classe que permite fazer brute force de dataset para identificar melhor amostra de treino, razão de True/False e melhores Hyperparametros.
 É composta por 3 funções construtoras e utiliza várias outras presentes dentro desse pacote.
 Para avaliar o modelo é possível usar função de scoring do próprio Sklearn ou criar o seu próprio parametro de avaliação. Para fazer isso, ver maiores informações na documentação do sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
 Dentro desse pacote temos a custom_confusion que foi feita para ser usada no maker_scorer do SKlearn

 Algumas explicações de parâmetros da classe:
 - **model:** Modelo a ser otimizado
 - **iterations**: Número de rodadas de iteração a serem realizadas, quanto mais rodadas mais tempo vai demorar!(exponencial)
 - **ratio**: ritmo de evolução da proporção True/False entre as iterações. Ex.: Iterações = 10, ratio = 1 então ao todo 10 tamanhos de amostras serão testadas e para cada uma dessas 10 amostras, 10 razões de proporção entre True/False vão ser testadas. Total de rodadas de otimização do exemplo: 100
 - **scorer**: Aceita 'recall', 'precision', 'f1' e etc.. Aceita função própria.
 - **regression**: A classe de sample_stressing, por padrão, foca em modelos de classificação porém é possível usar ela para modelos de regressão também.
 - **drop** : Lista de colunas a serem dropadas do dataset final, se não existir nenhuma só ignorar.

### Resample
Função usada pela classe SampleStress para gerar as amostras com diferentes tamanhos e razões de target
```python
def resample(dataframe, sample_ratio=1, sample_size=0):
    header = dataframe.columns
    positives = (dataframe.loc[dataframe['target']==1]).to_numpy()
    pos_ratio = int(sample_size/(sample_ratio+1))
    if sample_size>0:
        if pos_ratio < len(positives):
            positives = positives[:pos_ratio]
        else:
            iterations = int(pos_ratio/len(positives))
            r = int(((pos_ratio/len(positives)) % 1)*len(positives))
            temp = positives
            for i in range(iterations-1):
                temp = np.append(temp, positives, axis=0)
            positives = np.append(temp, positives[:r], axis=0)
        sample_size=len(positives)
    else:
        sample_size = len(positives)
    positives = pd.DataFrame(positives)
    positives.columns = header
    neg_ratio = sample_size*sample_ratio
    negatives = (dataframe.loc[dataframe['target']==0])[:neg_ratio].to_numpy()
    negatives = pd.DataFrame(negatives)
    negatives.columns = header
    return negatives.append(positives).reset_index().drop(columns=['index'])
```

### DataBuilder
Para datasets muito grandes, construí essa classe que tem duas funções que iteram pelo dataframe e fazem o tratamento dos dados para tornar o dataset mais leve e reduzir o consumo de memória ram. Isso acelera muito o tempo de manipulação dos dados pois impede que os dados vazem para armazenamento mais lento como HD/SSD

A classe ainda está "especializada" para o caso que foi construída, isso é rápido de ajeitar e é o próximo passo para ela. Por esse motivo ainda não vou colocar mais detalhes.
