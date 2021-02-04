import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tcn.tcn import TCN
from tensorflow import keras

window_size = 20   # 
batch_size = 32    # 
epochs = 200       # 
filter_nums = 10   # 
kernel_size = 4    # 


def get_dataset():
    df = pd.read_csv('./bars/AMZN Historical Data.csv', thousands=',')
#    df = df[::-1]
    scaler = MinMaxScaler()
    open_arr = scaler.fit_transform(df['Open'].values.reshape(-1, 1)).reshape(-1)
    X = np.zeros(shape=(len(open_arr) - window_size, window_size))
    label = np.zeros(shape=(len(open_arr) - window_size))
    for i in range(len(open_arr) - window_size):
        X[i, :] = open_arr[i:i+window_size]
        label[i] = open_arr[i+window_size]
    train_X = X[:2000, :]
    train_label = label[:2000]
    test_X = X[2000:3000, :]
    test_label = label[2000:3000]
    return train_X, train_label, test_X, test_label, scaler

def RMSE(pred, true):
    return np.mean(np.sqrt(np.square(pred - true)))

def plot(pred, true):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(pred)), pred)
    ax.plot(range(len(true)), true)
    plt.show()

def build_model():
    train_X, train_label, test_X, test_label, scaler = get_dataset()
    model = keras.models.Sequential([
        keras.layers.Input(shape=(window_size, 1)),
        TCN(nb_filters=filter_nums,                   
            kernel_size=kernel_size,                   
            dilations=[1, 2, 4, 8]),     
        keras.layers.Dense(units=1, activation='relu')
    ])
    model.summary()
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    model.fit(train_X, train_label, validation_split=0.2, epochs=epochs)

    model.evaluate(test_X, test_label)
    prediction = model.predict(test_X)
    scaled_prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(-1)
    scaled_test_label = scaler.inverse_transform(test_label.reshape(-1, 1)).reshape(-1)
    print('RMSE ', RMSE(scaled_prediction, scaled_test_label))
    plot(scaled_prediction, scaled_test_label)
