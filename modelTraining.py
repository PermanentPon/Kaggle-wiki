import pandas as pd
import dask.dataframe as dd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import multiprocessing

window_size = 60

def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    for i in range(0, len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size:i + window_size + 1])
    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    return X,y

def prepare_data(train):
    X_total = np.array([], dtype=np.int64).reshape(0, window_size)
    y_total = np.array([], dtype=np.int64).reshape(0, 1)
    for index, row in train.iloc[:, 1:-1].iterrows():
        X, y = window_transform_series(row, window_size)
        X_total = np.concatenate((X_total, X))
        y_total = np.concatenate((y_total, y))
    return X_total, y_total

def parallelization(train):
    # create as many processes as there are CPUs on your machine
    num_processes = multiprocessing.cpu_count()
    print("Number of processes: " + str(num_processes))
    # calculate the chunk size as an integer
    chunk_size = int(train.shape[0] / num_processes)
    chunks = [train.ix[train.index[i:i + chunk_size]] for i in range(0, train.shape[0], chunk_size)]
    pool = multiprocessing.Pool(processes=num_processes)
    return pool.map(prepare_data, chunks)

if __name__ == "__main__":
    print("Started data loading")
    train = pd.read_csv('train_1_clustered.csv').fillna(0)
    print(train.columns)
    grouped_train = train.groupby('cluster')
    for cluster, data in grouped_train:
        #cluster_ch = np.char.mod('%d', cluster)
        print("Started data preparation for cluster " + str(cluster))
        X, y = parallelization(data)

        # input must be reshaped to [samples, window size, stepsize]
        X = np.asarray(np.reshape(X, (X.shape[0], window_size, 1)))
        print("Finished data preparation for cluster " + str(cluster))

        prep_data = dd.DataFrame(X)
        prep_data[len(y)] = y

        prep_data.to_csv('Prep_data_' + str(cluster) + '.csv')

        model = Sequential()
        model.add(LSTM(128, input_shape=(window_size, 1)))
        model.add(Dense(1))

        # build model using keras documentation recommended optimizer initialization
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

        # compile the model
        model.compile(loss='mean_absolute_error', optimizer=optimizer)
        print("Started fitting model for cluster " + str(cluster))
        model.fit(X, y, epochs=100, batch_size=50, verbose=1)
        print("Finished fitting model for cluster " + str(cluster))
        model.save('./Models/wiki_model_LSTM_1:128/model_' + str(cluster) + ".h5")
        print("Saved model " + './Models/wiki_model_LSTM_1_128/model_' + str(cluster) + ".h5")
