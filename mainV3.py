from copy import deepcopy as copy
from datetime import datetime
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


def feature_engineering(df_raw, features, estrategy='simple'):
    if estrategy == '':
        pass
    else:
        df = df_raw[features + ['timestamp']]
        df = df.set_index('timestamp', drop=True)

    return df


def format_data(data_list, labels, janela_de_tempo, janela_de_predicao):
    n_features = len(data_list)
    n_dataset = len(data_list[0])
    hist_list = [[] for _ in range(n_features)]
    target = []

    for i in range(n_dataset - (janela_de_tempo + janela_de_predicao)):
        for data_index, data in enumerate(data_list):
            x = data[i:i + janela_de_tempo]
            hist_list[data_index].append(x)
        y = labels[i + janela_de_tempo:i + janela_de_tempo + janela_de_predicao]

        target.append(y)

    # convertendo de lista para array
    hist_list = [np.array(hist) for hist in hist_list]
    target = np.array(target)

    return hist_list, target


def split_train_test(hist_list, target):
    """
        60% train
        20% test
        20% validation
    """

    X_train_list, X_test_list, X_val_list = [], [], []

    for hist in hist_list:
        X_train, X_test, y_train, y_test = train_test_split(hist, target, test_size=0.2, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                                          random_state=1)  # 0.25 x 0.8 = 0.2

        X_train_list.append(X_train)
        X_test_list.append(X_test)
        X_val_list.append(X_val)

    return X_train_list, X_test_list, X_val_list, y_train, y_test, y_val


def normalize_data(X_train_list, X_test_list, X_val_list, y_train, y_test, y_val, janela_de_tempo):
    ##Normalizando...

    for index, (X_train, X_test, X_val) in enumerate(zip(X_train_list, X_test_list, X_val_list)):
        sc = MinMaxScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)
        X_test = sc.transform(X_val)

        X_train = X_train.reshape((len(X_train), janela_de_tempo, 1))
        X_test = X_test.reshape((len(X_test), janela_de_tempo, 1))
        X_val = X_test.reshape((len(X_val), janela_de_tempo, 1))

        X_train_list[index] = X_train
        X_test_list[index] = X_test
        X_val_list[index] = X_val

    X_train = np.dstack(X_train_list)
    X_test = np.dstack(X_test_list)
    X_val = np.dstack(X_val_list)

    sc.fit(y_train)
    y_train = sc.transform(y_train)
    y_test = sc.transform(y_test)
    y_val = sc.transform(y_val)

    return X_train, X_test, X_test, y_train, y_test, y_val, sc


def create_model(janela_de_tempo, y_train, features, estrategy='simple'):
    model = None

    if estrategy == '':
        pass
    else:
        model = tf.keras.Sequential()

        # encoder
        model.add(LSTM(25, input_shape=(janela_de_tempo, len(features))))
        model.add(Dropout(0.10))

        # Gate do decoder
        model.add(RepeatVector(y_train.shape[1]))

        # decoder
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
                estrategy='simple',
                optimizer='adam',
                loss='binary_crossentropy',
                janela_de_tempo=30,   # Quantidade de slots utilizados pra predicao
                janela_de_predicao=10,  # Quanditade de slots pra frente que serao preditos
                epochs=2,
                batch_size=32,
                features=['close']
            ),
        dict(
            estrategy='simple_02',
            optimizer='adam',
            loss='binary_crossentropy',
            janela_de_tempo=30,  # Quantidade de slots utilizados pra predicao
            janela_de_predicao=10,  # Quanditade de slots pra frente que serao preditos
            epochs=2,
            batch_size=32,
            features=["high", "low", "close"]
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
            features = parameter['features']

            file_path = os.path.join(path, file)
            df_raw = load_file(file_path)
            # del df_raw
            df = feature_engineering(
                df_raw,
                features,
                estrategy=estrategy
            )
            
            df.loc[:, 'rotulo'] = df['close']
            data_list = [df.loc[:, feature] for feature in features]
            labels = df['rotulo']

            hist_list, target = format_data(data_list, labels, janela_de_tempo, janela_de_predicao)
            dict_log['len_database'] = len(data_list[0])

            X_train_list, X_test_list, X_val_list, y_train, y_test, y_val = split_train_test(hist_list, target)
            X_train, X_test, X_val, y_train, y_test, y_val, sc = normalize_data(
                X_train_list, X_test_list, X_val_list, y_train, y_test, y_val, janela_de_tempo
            )

            model = create_model(janela_de_tempo, y_train, features, estrategy='simple')
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size)

            pred = model.predict(X_test)

            mae_val = mean_absolute_error([i[0] for i in y_test], [i[0] for i in pred])
            dict_log['history'] = history.history
            dict_log['mae_val'] = mae_val


            log_list.append(dict_log)

            now_ts = int(datetime.now().timestamp())
            pd.DataFrame(log_list).to_csv(f'results/resultados-v2-{now_ts}.csv')


if __name__ == "__main__":
    main()
