# motor_anomaly_detection
The content of this repository presents an intelligent fault detection system for electric motors. Vibration (X-Axis Acc, Y-Axis Acc, Z-Axis Acc) and current consumption data was collected from a Nema17 stepper motor, and stored in the all_motor_data.csv file. The vibration data was collected using an accelerometer sensor. After all the data collection is complete, the fault or anomaly detection model is trained using the code in model_training.py file. The information on how the model is deployed and what the results are can be found in keras_model_anomaly.ipynb file.

The anomaly detection model has and autoencoder structure. It is trained on regular signal data. As an autoencoder, the model is able to reconstruct familiar input signal. Therefore, if there is an issue with the motor, the error between the input and reconstructed signal would exceed acceptable thresholds. Status of the motor can be viewed on a web user interface.

<p align="center">
  <img src="Autoencoder Model.PNG" width="600">
</p>

**Figure 1:** Autoencode-based fault detection model.
