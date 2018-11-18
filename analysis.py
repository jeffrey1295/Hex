## parameter setting
x_path = 'Data/X_all.csv'
y_path = ['Data/Y_Fphi.csv','Data/Y_Ftheta.csv','Data/Y_Tphi.csv']
m_path = 'Result/test01.h5'
p_path = 'Result/test01.npz'
t_path = 'Result/logs/'
ns = 6
n1 = 64; w1 = 'zeros'; b1 = 'zeros'; a1 = 'tanh'; r1 = 0.0; d1 = 0.5  # w='glorot_uniform'
nf = 3
l = 'mean_squared_error'
lr = 0.001; e = 20; b = 200 # ~180 data per exp
v = 0.16 # ~6

## read data
print('Reading data')
import numpy as np
import pandas as pd
x = pd.read_csv(x_path,header=None)
y = pd.Series()
for i,p in enumerate(y_path):
    y = pd.concat([y,pd.read_csv(p,header=None)],axis=1)
y = y.iloc[:,1:]
y.columns = range(len(y_path))

## statistical analysis
print('Analizing')
from matplotlib import pyplot as plt
xy = pd.concat([x,y],axis=1)
Correlation = xy.corr()
x_mean = x.mean(axis=0)
x_std = x.std(axis=0)
x = (x-x_mean)/x_std
y_mean = y.mean(axis=0)
y_std = y.std(axis=0)
y = (y-y_mean)/y_std
plt.figure(1); plt.plot(x); plt.title('x (zscore)')
plt.figure(2); plt.plot(y); plt.title('y (zscore)')

## preprocessing

## training
print('Training')
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
model = Sequential()
model.add(Dense(
        batch_input_shape=(None,ns),
        units = n1,
        kernel_initializer=w1,
        bias_initializer=b1,
        activation = a1,
        kernel_regularizer=regularizers.l2(r1),        
        ))
model.add(Dropout(d1))
model.add(Dense(nf))
opt = Adam(lr=lr)
model.compile(optimizer=opt,loss=l)
Loss = model.fit(
        x,y,
        batch_size=b,
        epochs=e,
        verbose=1,
        callbacks=[TensorBoard(log_dir=t_path)],
        validation_split=0.2,
        #validation_data=,
        )

## result
print('Ploting results')
plt.figure(3);
plt.plot(Loss.history['loss']); plt.plot(Loss.history['val_loss']);
plt.title('loss / valid_loss (mean square error) (zscore used)');
Prediction = model.predict(x)
for i in range(len(y_path)):
    plt.figure(4+i);
    plt.plot(y[i]); plt.plot(Prediction[:,i]);
pxy = pd.concat([y,pd.DataFrame(Prediction)],axis=1)
Correlation_of_prediction = pxy.corr()

## saving
print('Saving')
model.save(m_path)
np.savez(p_path,
        Correlation = Correlation,
        Mean = pd.concat([x_mean,y_mean]),
        Std = pd.concat([x_std,y_std]),
        Parameters = {'lr':lr,'e':e,'b':b,'v':v},
        Loss = Loss.history,
        Prediction = Prediction,
        Correlation_of_prediction = Correlation_of_prediction,
        )