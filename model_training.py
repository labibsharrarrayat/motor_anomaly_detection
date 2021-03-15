# lstm autoencoder to recreate a timeseries
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
'''
A UDF to convert input data into 3-D
array as required for LSTM network.
'''
df = pd.read_csv('all_motor_data.csv',header=None)
def temporalize(X, y, lookback):
    output_X = []
    output_y = []
    for i in range(len(X)-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            t.append(X[[(i+j+1)], :])
        output_X.append(t)
        output_y.append(y[i+lookback+1])
    return output_X, output_y

# define input timeseries
timeseries = np.array([df[0].values,df[1].values,df[2].values,df[3].values]).transpose()

print(timeseries)

timesteps = 10
X, y = temporalize(X = timeseries, y = np.zeros(len(timeseries)), lookback = timesteps)

n_features = 4
X = np.array(X)
X = X.reshape(X.shape[0], timesteps, n_features)

print(X)
print(X.shape)

# define model
model = Sequential()
model.add(LSTM(256, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=False))
model.add(RepeatVector(timesteps))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(LSTM(256, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', metrics=['accuracy'], loss='mse')
model.summary()

# checkpoint
filepath="D:/Users/User/Documents/MOTOR_FAULT_Models/anomaly_model/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = model.fit(X, X, validation_split=0.33, epochs=100, batch_size=10, callbacks=callbacks_list, verbose=1)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()