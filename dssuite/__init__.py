import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import time
import datetime

name="ds-suite/ds-suite"
__version__ = "1.0.5"

#-------------
def to_date(gamma):
    #print(gamma[gamma['created_at']==erro])
    g = pd.isnull(pd.to_datetime(gamma['created_at'], errors='coerce')).values
    gamma = gamma[~g]
    gamma['created_at'] = pd.to_datetime(gamma['created_at'])
    return gamma


# In[337]:


#@profile
def jit_to_column(dataframe):
    month = []
    day = []
    hour = []
    for row in range(len(dataframe)):
            month.append(dataframe[row].astype('datetime64[M]').astype(int) % 12 + 1)
            day.append(dataframe[row].astype('datetime64[D]').astype(int) % 31 + 1)
            hour.append(dataframe[row].astype('datetime64[h]').astype(int) % 24 + 1)
    return month, day, hour


# In[ ]:

def numberfy(dataframe, target, string):
    num = dataframe[target].to_numpy()
    for i in range(len(num)):
        if num[i] == string:
            num[i] = 1.0
        else:
            num[i] = 0.0
    dataframe[target]=num

# In[ ]:
def confusion(predict, test_set, theresold):
    return_matrix = {'T':0,'R':0, 'P':0, 'O':0, 'TP':0, 'FP':0}
    predict = np.array([1 if i >= theresold else 0 for i in predict])
    test_set = test_set.to_numpy().astype(int)
    confusion = cm(test_set, predict)
    length = len(test_set)
    condition_P = confusion[1].sum()
    True_P = confusion[1][1]
    False_P = confusion[0][1]
    prediction_P = predict.sum()
    #====
    Recall = round((True_P/condition_P)*100, 2)
    Out_of_total = round(((False_P)/length)*100, 2)
    Precision  = round(True_P/prediction_P*100, 2)
    return_matrix['T'] = theresold
    return_matrix['R'] = Recall
    return_matrix['P'] = Precision
    return_matrix['O'] = Out_of_total
    return_matrix['FP'] = False_P
    return_matrix['TP'] = True_P
    return return_matrix

def simple_confusion_legacy(predict, test_set, theresold):
    return_matrix = {'T':0,'R':0, 'P':0, 'O':0, 'TP':0, 'FP':0, 'CP':0}
    length = len(test_set.to_list())
    condition_P = test_set.to_list().count(1.0)
    test_set=test_set.to_numpy()
    True_P = len([i for i, j in zip(predict, test_set) if j == 1.0 and i>=theresold])
    False_P = len([i for i, j in zip(predict, test_set) if j == 0.0 and i>=theresold])
    False_N = len([i for i, j in zip(predict, test_set) if j == 1.0 and i<theresold])
    True_N = len([i for i, j in zip(predict, test_set) if j == 0.0 and i<theresold])
    prediction_P = len([i for i in range(len(predict)) if predict[i]>=theresold])
    #====
    if prediction_P>0:
        Precision  = round((True_P/prediction_P)*100, 2)
    else:
        Precision = 0.0
    Recall = round((True_P/condition_P)*100, 2)
    class_acc = round(((False_P)/length)*100, 2)
    return_matrix['T'] = theresold
    return_matrix['R'] = Recall
    return_matrix['P'] = Precision
    return_matrix['CA'] = class_acc
    return_matrix['FP'] = False_P
    return_matrix['TP'] = True_P
    return_matrix['CP'] = condition_P
    return return_matrix

def simple_confusion(predict, test_set):
    return_matrix = {'T':0,'R':0, 'P':0, 'O':0, 'TP':0, 'FP':0}
    top_1_percent = np.array(predict).argsort()[int(len(predict)*0.99):][::-1]
    threshold = np.unique(np.array(predict)[top_1_percent])[1]
    top_1_percent = np.array(predict)[top_1_percent]
    predict = np.array([1 if i >= threshold else 0 for i in predict])
    test_set = test_set.to_numpy().astype(int)
    confusion = cm(test_set, predict)
    length = len(test_set)
    condition_P = confusion[1].sum()
    True_P = confusion[1][1]
    False_P = confusion[0][1]
    prediction_P = predict.sum()
    #====
    Recall = round((True_P/condition_P)*100, 2)
    Out_of_total = round(((False_P+True_P)/length)*100, 2)
    Precision  = round(True_P/prediction_P*100, 2)
    return_matrix['T'] = min(top_1_percent)
    return_matrix['R'] = Recall
    return_matrix['P'] = Precision
    return_matrix['O'] = Out_of_total
    return_matrix['FP'] = False_P
    return_matrix['TP'] = True_P
    return return_matrix

def custom_confusion(test_set, predict):
    test_set=test_set.to_list()
    length = sorted(range(len(predict)), key=lambda i: predict[i], reverse=True)[:int(len(predict)*0.01)]
    pred = predict[length[len(length)-1]]
    theresold = pred
    True_P = len([i for i, j in zip(predict, test_set) if j == 1.0 and i>=theresold])
    prediction_P = len([[].append(item) for item in predict if item>=theresold])
    return prediction_P - True_P


# In[ ]:


def read_confusion_output(output):
    if isinstance(output['T'], list):
        for i in range(len(output['T'])):
            print("=================================")
            print("For the theresold: "+'>'+str(round(output['T'][i]*100,2))+'%')
            print("Recall: "+output['R'][i])
            print("Precision: "+output['P'][i])
            print("Perda de Conversao: "+output['O'][i])
            print("=================================")
    else:
        print("=================================")
        print("For the theresold: "+'>'+str(round(output['T']*100,2))+'%')
        print("Recall: "+str(output['R']))
        print("Precision: "+str(output['P']))
        print("Perda de Conversao: "+str(output['O']))
        print("False Positives: "+str(output['FP']))
        print("True Positives: "+str(output['TP']))
        print("=================================")

def read_stressing(output, best=True):
    place = len([i for i in range(len(output['T'])) if len(output['T'][i])>=2])
    if  best==True:
        temp = {'Size':[], 'Ratio':[], 'Recall':[], 'Precision':[], 'True +':[], 'False +':[], 'Block':[], 'Threshold':[],'Optimal Model':[]}
        if len(output['S']) <= 1:
            for item in range(len(output['R'][0])):
                temp['Size'].append(output['S'][0])
                temp['Ratio'].append(output['RT'][0][item])
                temp['Recall'].append(output['R'][0][item])
                temp['Precision'].append(output['P'][0][item])
                temp['True +'].append(output['TP'][0][item])
                temp['False +'].append(output['FP'][0][item])
                temp['Block'].append(output['O'][0][item])
                temp['Optimal Model'].append(output['model'][0][item])
                temp['Threshold'].append(output['T'][0][item])
            return pd.DataFrame(temp).sort_values(['Recall'], ascending=False)
        else:
            recall = np.array(output['R'][:place])
            for item in range(len(output['S'][:place])):
                index = recall[item].argsort()[::-1][0]
                temp['Size'].append(output['S'][item])
                temp['Ratio'].append(output['RT'][item][index])
                temp['Recall'].append(output['R'][item][index])
                temp['Precision'].append(output['P'][item][index])
                temp['True +'].append(output['TP'][item][index])
                temp['False +'].append(output['FP'][item][index])
                temp['Block'].append(output['O'][item][index])
                temp['Optimal Model'].append(output['model'][item][index])
                temp['Threshold'].append(output['T'][item][index])
            return pd.DataFrame(temp).sort_values(['Recall'], ascending=False)
    else:
        return pd.DataFrame({'Size':output['S'], 'Ratio':output['RT'], 'Recall':output['R'],
              'Precision':output['P'], 'True +':output['TP'], 'False +':output['FP'], 'Block':output['O'], 'Threshold':output['T'],'Optimal Model':output['model']})

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
    #-----------------------------------------------
    neg_ratio = sample_size*sample_ratio
    negatives = (dataframe.loc[dataframe['target']==0])[:neg_ratio].to_numpy()
    negatives = pd.DataFrame(negatives)
    negatives.columns = header
    del dataframe
    return negatives.append(positives).reset_index().drop(columns=['index'])

# In[SampleStress]:
# Classe que permite fazer brute force de dataset para identificar melhor amostra de treino, razão de True/False e melhores Hyperparametros.
# É composta por 3 funções construtoras e utiliza várias outras presentes dentro desse pacote.
# Para avaliar o modelo é possível usar função de scoring do próprio Sklearn ou criar o seu próprio parametro de avaliação. Para fazer isso, ver maiores informações na documentação do sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
# Dentro desse pacote temos a custom_confusion que foi feita para ser usada no maker_scorer do SKlearn
# Algumas explicações de parâmetros da classe:
# - Model: Modelo a ser otimizado
# - Iterations: Número de rodadas de iteração a serem realizadas, quanto mais rodadas mais tempo vai demorar!(exponencial)
# - ratio: ritmo de evolução da proporção True/False entre as iterações. Ex.: Iterações = 10, ratio = 1 então ao todo 10 tamanhos de amostras serão testadas e para cada uma dessas 10 amostras, 10 razões de proporção entre True/False vão ser testadas. Total de rodadas de otimização do exemplo: 100
# - scorer: Aceita 'recall', 'precision', 'f1' e etc.. Aceita função própria.
# - regression: A classe de sample_stressing, por padrão, foca em modelos de classificação porém é possível usar ela para modelos de regressão também.
# - drop : Lista de colunas a serem dropadas do dataset final, se não existir nenhuma só ignorar.


class sample_stress:
    def __init__(self, train, test, model, iterations, ratio, drop, criteria='P', scorer='recall', regression=False, sizes = None, limit=float("inf")):
        self.train = train
        self.test = test
        self.model = model
        self.scorer = scorer
        self.iterations = iterations
        self.ratio = ratio
        self.logs = {'S':[], 'RT':[],'R':[], 'P':[], 'TP':[], 'FP':[], 'O':[], 'model':[], 'T':[]}
        self.sizes = []
        self.db = {'x':[], 'y':[]}
        self.regression = regression
        self.drop = drop
        self.sizes = sizes
        self.limit = limit
        self.criteria = criteria

    def explain():
        print("""Classe que permite fazer brute force de dataset para identificar melhor amostra de treino, razão de True/False e melhores \nHyperparametros.\nÉ composta por 3 funções construtoras e utiliza várias outras presentes dentro desse pacote.\nPara avaliar o modelo é possível usar função de scoring do próprio Sklearn ou criar o seu próprio parametro de avaliação. Para fazer isso, ver maiores informações na documentação do sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html\nDentro desse pacote temos a custom_confusion que foi feita para ser usada no maker_scorer do SKlearn\nAlgumas explicações de parâmetros da classe:\n    - Model: Modelo a ser otimizado\n    - Iterations: Número de rodadas de iteração a serem realizadas, quanto mais rodadas mais tempo vai demorar!(exponencial)\n    - ratio: ritmo de evolução da proporção True/False entre as iterações. Ex.: Iterações = 10, ratio = 1 então ao todo 10 tamanhos de amostras serão testadas e para cada uma dessas 10 amostras, 10 razões de proporção entre True/False vão ser testadas. Total de rodadas de otimização do exemplo: 100\n    - scorer: Aceita 'recall', 'precision', 'f1' e etc.. Aceita função própria.\n    - regression: A classe de sample_stressing, por padrão, foca em modelos de classificação porém é possível usar ela para modelos de regressão também.\n    - drop : Lista de colunas a serem dropadas do dataset final, se não existir nenhuma só ignorar. \n    - sizes: Lista de amostras para serem usadas\n    - limit: Limite de tentativas logadas(default=Infinito)""")

    def samples(self, one_sample=False, sample=5000):
        if one_sample==False:
            if type(self.sizes) != list:
                self.sizes = [5000, 7500, 10000, 12500, 15000, 17500, 20000, 25000, 30000, 50000]
            if len(self.sizes) < self.iterations:
                for r in range(self.iterations-len(self.sizes)):
                    temp = np.random.randint(50, 500)*1000
                    if temp not in self.sizes:
                        self.sizes.append(temp)
            for i in range(self.iterations):
                self.logs['S'].append(self.sizes[i])
                self.logs['RT'].append([])
                self.logs['R'].append([])
                self.logs['P'].append([])
                self.logs['TP'].append([])
                self.logs['FP'].append([])
                self.logs['O'].append([])
                self.logs['T'].append([])
                self.logs['model'].append([])
                self.db['x'].append([])
                self.db['y'].append([])
        else:
            self.sizes.append(sample)
            self.logs['S'].append(sample)
            self.logs['RT'].append([])
            self.logs['R'].append([])
            self.logs['P'].append([])
            self.logs['TP'].append([])
            self.logs['FP'].append([])
            self.logs['O'].append([])
            self.logs['T'].append([])
            self.logs['model'].append([])
            self.db['x'].append([])
            self.db['y'].append([])

    def ratio_stress(self, size, ids):
        l = self.logs['S'].index(size)
        for i in range(self.iterations):
            print("==========================")
            start_time = datetime.datetime.now()
            print(f'Iterando amostra: {size} | ID: {ids} | Ratio: {self.ratio*(i+1)}:1')
            temp = resample(self.train, int(self.ratio*(i+1)), size)
            print(f'Resample: {datetime.datetime.now()-start_time}')
            start_time = datetime.datetime.now()
            y_train = temp['target'].reset_index()['target'].astype('float')
            x_train = temp.drop(columns=self.drop).dropna(how='any',axis=0).reset_index().drop(columns=['index'])
            print(f'Variable Setting: {datetime.datetime.now()-start_time}')
            start_time = datetime.datetime.now()
            model = self.model
            model.fit(x_train, y_train)
            print(f'Fit: {datetime.datetime.now()-start_time}')
            start_time = datetime.datetime.now()
            if self.regression==True:
                predict = model.predict(self.test['x'])
                print(f'Predict: {datetime.datetime.now()-start_time}')
                start_time = datetime.datetime.now()
                m_temp = simple_confusion(predict, self.test['y'])
            else:
                predict = model.predict_proba(self.test['x'])
                print(f'Predict: {datetime.datetime.now()-start_time}')
                start_time = datetime.datetime.now()
                m_temp = simple_confusion(predict[:,1], self.test['y'])
            if len(self.logs[self.criteria][l]) < self.limit:
                self.logs['RT'][l].append(self.ratio*(i+1))
                self.logs['R'][l].append(m_temp['R'])
                self.logs['P'][l].append(m_temp['P'])
                self.logs['TP'][l].append(m_temp['TP'])
                self.logs['FP'][l].append(m_temp['FP'])
                self.logs['O'][l].append(m_temp['O'])
                self.logs['T'][l].append(m_temp['T'])
                self.db['x'][l].append(x_train)
                self.db['y'][l].append(y_train)
                self.logs['model'][l].append(model)
            elif m_temp[self.criteria] > min(self.logs[self.criteria][l]):
                loc = self.logs[self.criteria][l].index(min(self.logs[self.criteria][l]))
                self.logs['RT'][l][loc] = self.ratio*(i+1)
                self.logs['R'][l][loc] = m_temp['R']
                self.logs['P'][l][loc] = m_temp['P']
                self.logs['TP'][l][loc] = m_temp['TP']
                self.logs['FP'][l][loc] = m_temp['FP']
                self.logs['O'][l][loc] = m_temp['O']
                self.logs['T'][l][loc] = m_temp['T']
                self.db['x'][l][loc] = x_train
                self.db['y'][l][loc] = y_train
                self.logs['model'][l][loc] = model
            print(f'Confuse: {datetime.datetime.now()-start_time}')
    def grid_stress(self, size, stress):
        CV_rfc = GridSearchCV(self.model, param_grid={'n_estimators':[50,100,200,300,500],'max_depth':[3,6,9,12],'booster':['gbtree','gblinear', 'dart'], 'learning_rate':[0.1,0.25,0.5,0.75]}, cv=5, scoring=self.scorer, n_jobs=-1)
        l = self.logs['S'].index(size)
        for i in range(len(self.db['x'][l])):
            print("==========================")
            print("==========================")
            print('GridSearch™ing Amostra: '+str(size)+' | ID: '+str(i)+' | Ratio: '+str(self.logs['RT'][l][i]))
            print("==========================")
            start_time = time.time()
            CV_rfc.fit(self.db['x'][l][i], self.db['y'][l][i])
            print('GridSearch™ Fit: '+time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))
            start_time = time.time()
            model = CV_rfc.best_estimator_
            predict = model.predict_proba(self.test['x'])
            print('GridSearch™ Predict: '+time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))
            start_time = time.time()
            m_temp = simple_confusion(predict[:,1], self.test['y'])
            print('GridSearch™ Confusion: '+str(m_temp['R'])+' | '+time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))
            if m_temp['R'] > min(self.logs['R'][l]):
                loc = self.logs['R'][l].index(min(self.logs['R'][l]))
                if m_temp['R'] > self.logs['P'][l][loc]:
                    self.logs['RT'][l][loc] = self.ratio*(i+1)
                    self.logs['R'][l][loc] = m_temp['R']
                    self.logs['P'][l][loc] = m_temp['P']
                    self.logs['TP'][l][loc] = m_temp['TP']
                    self.logs['FP'][l][loc] = m_temp['FP']
                    self.logs['O'][l][loc] = m_temp['O']
                    self.logs['T'][l][loc] = m_temp['T']
                    self.logs['model'][l][loc] = CV_rfc.best_estimator_

    def clean_run(self, obj, one_sample=False, stress=[]):
        obj.samples(one_sample=one_sample)
        if one_sample==False:
            ids = 0
            for i in range(obj.iterations):
                obj.ratio_stress(obj.sizes[i], ids)
                ids += 1
            if len(stress)>0:
                ids=0
                for i in range(obj.iterations):
                    obj.grid_stress(obj.sizes[i], stress)
                    ids+=1
        else:
            obj.ratio_stress(obj.sizes[0], ids)
            if len(stress)>0:
                obj.grid_stress(obj.sizes[0], stress)

# Para datasets muito grandes, construí essa classe que tem duas funções que iteram pelo dataframe e fazem o tratamento dos dados para tornar o dataset mais leve e reduzir o consumo de memória ram. Isso acelera muito o tempo de manipulação dos dados pois impede que os dados vazem para armazenamento mais lento como HD/SSD

#A classe ainda está "especializada" para o caso que foi construída, isso é rápido de ajeitar e é o próximo passo para ela.

class data_builder:
    def __init__(self, path, chunk_size):
        self.path = path
        self.chunk_size = chunk_size
        self.log = {'chunk_id':[], 'time':[]}
        self.output = ['゠','゠','゠','゠','゠',
                       '゠','゠','゠','゠','゠',
                       '゠','゠','゠','゠','゠',
                       '゠','゠','゠','゠','゠',
                       '゠']

    def result_bar(self, end_goal, current_step, step_name):
        progress = int(round((current_step/end_goal)*20, 0))
        self.output[progress] = '▇'
        #clear_output(wait=True)
        print(step_name)
        print(' ')
        print("Progress({}%):".format(progress*5))
        print('[{}]'.format(''.join(self.output[1:])))

    def iter_chunks(self, date_col, subset, logging=False):
        self.output = ['゠','゠','゠','゠','゠',
                       '゠','゠','゠','゠','゠',
                       '゠','゠','゠','゠','゠',
                       '゠','゠','゠','゠','゠',
                       '゠']
        iter_csv = pd.read_csv(self.path, iterator=True, chunksize=self.chunk_size)
        print('Starting..... Usually this step takes 2-3 minutes')
        l = [1 for chunk in iter_csv]
        l = len(l)
        iter_csv = pd.read_csv(self.path, iterator=True, chunksize=self.chunk_size)
        dfs = [pd.DataFrame({})]
        c = 0
        last_batch = 0
        start_time = datetime.datetime.now()
        for chunk in iter_csv:
            start_time_chunk = datetime.datetime.now()
            chunk = chunk.convert_dtypes()
            chunk['anomes']=chunk.loc[~chunk[date_col].isna()][date_col].apply(lambda x: int(x[:10].replace('-','')))
            chunk = chunk.dropna(subset=subset)
            chunk = chunk.reset_index().drop(columns=['index'])
            temp = chunk.groupby('msisdnrecebedor').min().reset_index()[['msisdnrecebedor', 'datahoratransacao']]
            temp = temp.rename(columns = {'datahoratransacao':'datahoraadesao'})
            chunk['datahoraadesao'] = pd.merge(how='left', on='msisdnrecebedor',
                left=chunk,
                right=temp)['datahoraadesao']
            if int(c/100)!=last_batch:
                dfs.append(pd.DataFrame({}))
                last_batch = int(c/100)
            dfs[last_batch] = pd.concat([dfs[last_batch], chunk], ignore_index=True)
            elapsed_time_chunk = datetime.datetime.now()
            if logging == True:
                self.log['chunk_id'].append(c)
                self.log['time'].append(elapsed_time_chunk-start_time_chunk)
            c+=1
            self.result_bar(l, c, 'Iterating through chunks')
        print(datetime.datetime.now()-start_time)
        return dfs

    def concat_all(self, dfs):
        self.output = ['゠','゠','゠','゠','゠',
                       '゠','゠','゠','゠','゠',
                       '゠','゠','゠','゠','゠',
                       '゠','゠','゠','゠','゠',
                       '゠']
        dataframe = pd.DataFrame({})
        l = len(dfs)
        c = 0
        for df in dfs:
            dataframe = pd.concat([dataframe, df], ignore_index=True)
            del df
            c+=1
            self.result_bar(l, c, 'Joining parts into single dataframe')
        del dfs
        return dataframe
