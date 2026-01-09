from time import sleep
import time
import serial
import re
import pandas as pd
import numpy as np
from keras.models import load_model

ser = serial.Serial("COM4",9600)

x_val = []; y_val = []; z_val = [];
timesteps = 10
n_features = 3

threshold_x = 0.07000363134940327; threshold_y = 0.10376056537219391; threshold_z = 0.062183414039339524

Ax = 0; Ay = 0; Az = 0
anomaly = 0


model = load_model('D:/Users/User/Documents/MOTOR_FAULT_Models/anomaly_model13/weights-improvement-68-0.01.hdf5')


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

def reconstruction(arr_input):
    return model.predict(arr_input, verbose=1)


def mse_calc(df_real, df_recon):
    anomaly_x_pred = anomaly_y_pred = anomaly_z_pred = np.zeros(shape=(df_recon.shape[0], df_recon.shape[1], 1))
    anomaly_x = anomaly_y = anomaly_z = np.zeros(shape=(df_recon.shape[0], df_recon.shape[1], 1))

    for i in range(0, df_recon.shape[0]):
        for j in range(0, df_recon.shape[1]):
            anomaly_x_pred[i][j] = df_recon[i][j][0]
            anomaly_y_pred[i][j] = df_recon[i][j][1]
            anomaly_z_pred[i][j] = df_recon[i][j][2]

            anomaly_x[i][j] = df_real[i][j][0]
            anomaly_y[i][j] = df_real[i][j][1]
            anomaly_z[i][j] = df_real[i][j][2]


    test_mse_loss_x = (np.square(anomaly_x_pred - anomaly_x)).mean(axis=1)
    test_mse_loss_y = (np.square(anomaly_y_pred - anomaly_y)).mean(axis=1)
    test_mse_loss_z = (np.square(anomaly_z_pred - anomaly_z)).mean(axis=1)

    a_x = test_mse_loss_x[test_mse_loss_x > threshold_x].shape[0]
    a_y = test_mse_loss_y[test_mse_loss_y > threshold_y].shape[0]
    a_z = test_mse_loss_z[test_mse_loss_z > threshold_z].shape[0]



    #print(a_x)
    #print(a_y)
    #print(a_z)
    #print(a_amp)

    #return a_x, a_y, a_z, a_amp
    return a_x + a_y + a_z


while True:
    get_val = str(ser.readline())
    get_val = re.findall(r"[-+]?\d*\.\d+|\d+", get_val)

    if(len(get_val)==4):
        x_val.append(float(get_val[0])); y_val.append(float(get_val[1])); z_val.append(float(get_val[2]))

        if (len(x_val) >= 20 and len(y_val) >= 20 and len(z_val) >= 20):
            print("hold on")
            timeseries = np.array([np.array(x_val), np.array(y_val), np.array(z_val)]).transpose()

            df_in, df_out = temporalize(X=timeseries, y=np.zeros(len(timeseries)), lookback=timesteps)
            df_in = np.array(df_in)
            df_in = df_in.reshape(df_in.shape[0], timesteps, n_features)
            #print(df_in)

            df_re = reconstruction(df_in)

            #Ax, Ay, Az, Ac = mse_calc(df_in, df_re)
            anomaly += mse_calc(df_in, df_re)
            print(anomaly)

            #check_array(df_in)


            x_val = [];y_val = [];z_val = []



