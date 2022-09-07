from copy import deepcopy as copy
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint


path = 'data'


def load_file(file_path):
    
    return pd.read_csv(file_path)

def feature_engineering(df_raw, estrategy='simple'):
    
    if estrategy == 'simple':
        df = df_raw[['timestamp', 'close']]
        df = df.set_index('timestamp', drop=True)
        
    return df

def format_data(data, labels, janela_de_tempo, janela_de_predicao):
    hist = []
    target = []

    for i in range(len(data)-(janela_de_tempo + janela_de_predicao)):
        x = data[i:i+janela_de_tempo]
        y = labels[i+janela_de_tempo:i+janela_de_tempo+janela_de_predicao]

        hist.append(x)
        target.append(y)
    #convertendo de lista para array
    hist = np.array(hist)
    target = np.array(target)
    
    return hist, target

def split_train_test(hist, target):
    """
        60% train
        20% test
        20% validation
    """
    
    X_train, X_test, y_train, y_test = train_test_split(hist, target, test_size=0.2, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    
    return X_train, X_test, X_val, y_train, y_test, y_val

def normalize_data(X_train, X_test, X_val, y_train, y_test, y_val, janela_de_tempo):
    ##Normalizando...

    sc = MinMaxScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    X_test = sc.transform(X_val)

    X_train = X_train.reshape((len(X_train), janela_de_tempo, 1))
    X_test = X_test.reshape((len(X_test), janela_de_tempo, 1))
    X_val = X_test.reshape((len(X_val), janela_de_tempo, 1))
    
    sc.fit(y_train)
    y_train = sc.transform(y_train)
    y_test = sc.transform(y_test)
    y_val = sc.transform(y_val)
    
    return X_train, X_test, X_test, y_train, y_test, y_val, sc


def create_model(janela_de_tempo, y_train, estrategy='simple'):
    
    model = None
    
    if estrategy=='simple' :
        model = tf.keras.Sequential()

        #encoder
        model.add(LSTM(25, input_shape=(janela_de_tempo, 1)))
        model.add(Dropout(0.10))

        #Gate do decoder
        model.add(RepeatVector(y_train.shape[1]))

        #decoder
        model.add(LSTM(25, return_sequences=True))
        model.add(TimeDistributed(Dense(10)))
        model.add(Dense(1))

        print(model.summary())
    return model

def main():
    log_list = []

    list_files = os.listdir(path)
    #list_files = ['BCHUSDT-5m-data.csv', "ETCUSDT-5m-data.csv"]

    parameters_list = [
        dict(
            estrategy = 'simple',
            optimizer='adam',
            loss='binary_crossentropy',
            janela_de_tempo=60,   # Quantidade de slots utilizados pra predicao
            janela_de_predicao=10,  # Quanditade de slots pra frente que serao preditos
            epochs=2,
            batch_size=32
            
        )
    ]


    print(f'Existem {len(list_files)} arquivos')

    for file in list_files[:2]:

        for index_parameter, parameter in enumerate(parameters_list):
            print("----------------------------------------------------------")
            print(f'EXPERIMENTANDO: {file} ---- estrategia {index_parameter}')
            print(parameter)
            dict_log = copy(parameter)
            dict_log['index_parameter'] = index_parameter
            dict_log['file'] = file

            # Parametros do experimento
            estrategy = parameter['estrategy']
            optimizer=parameter['optimizer']
            loss=parameter['loss']
            janela_de_tempo =parameter['janela_de_tempo']
            janela_de_predicao =parameter['janela_de_predicao']
            epochs =parameter['epochs']
            batch_size =parameter['batch_size']

            file_path = os.path.join(path,file)
            df_raw = load_file(file_path)
            #del df_raw
            df = feature_engineering(
                df_raw,
                estrategy=estrategy
            )


            df.loc[:, 'rotulo'] = df['close']
            data = df.iloc[:, 0]
            labels = df['rotulo']

            hist, target = format_data(data, labels, janela_de_tempo, janela_de_predicao)
            dict_log['len_database'] = len(hist)

            X_train, X_test, X_val, y_train, y_test, y_val = split_train_test(hist, target)
            X_train, X_test, X_val, y_train, y_test, y_val, sc = normalize_data(
                X_train, X_test, X_val, y_train, y_test, y_val, janela_de_tempo
            )

            model = create_model(janela_de_tempo, y_train, estrategy=estrategy)
            model.compile(optimizer=optimizer, loss=loss)
            history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val),  batch_size=batch_size)

            pred = model.predict(X_test)

            mae_val = mean_absolute_error([i[0] for i in y_test], [i[0] for i in pred])
            dict_log['history'] = history.history
            dict_log['mae_val'] = mae_val

            log_list.append(dict_log)



            from datetime import datetime

            now_ts  = int(datetime.now().timestamp())
            pd.DataFrame(log_list).to_csv(f'results/resultados-v2-{now_ts}.csv')


if __name__ == "__main__":
    main()