import pandas as pd
import matplotlib.pyplot as plt
import scipy
import sys  
from scipy import signal
from scipy import pi
import numpy as np    
from scipy.signal import butter, lfilter, freqz
import os

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

fileName = input("Which file do you want?: ")

df = pd.read_csv(os.getcwd()+"/Data/"+fileName+".csv")

time = df['time'].to_list()
accx = df['accelerometerX'].to_list()
accy = df['accelerometerZ'].to_list()
accz = df['accelerometerY'].to_list()
gyrox = df['gyroscopeX'].to_list()
gyroy = df['gyroscopeY'].to_list()
gyroz = df['gyroscopeZ'].to_list()

#Centers the data at 0
accx2 = butter_highpass_filter(accx, 0.2, 25)
accy2 = butter_highpass_filter(accy, 0.2, 25)
accz2 = butter_highpass_filter(accz, 0.2, 25) 
gyrox2 = butter_highpass_filter(gyrox, 0.05, 25)
gyroy2 = butter_highpass_filter(gyroy, 0.05, 25)
gyroz2 = butter_highpass_filter(gyroz, 0.05, 25) 

#Movement thresholds
accThreshold = 300
gyrothreshold = 5000

#Basic movement detection
accxMovement = [10000 if accx2[i] >= accThreshold else np.nan for i in range(len(accx2))]
accyMovement = [10000 if accy2[i] >= accThreshold else np.nan for i in range(len(accy2))]
acczMovement = [10000 if accz2[i] >= accThreshold else np.nan for i in range(len(accz2))]
gyroxMovement = [10000 if gyrox2[i] >= gyrothreshold else np.nan for i in range(len(gyrox))]
gyroyMovement = [10000 if gyroy2[i] >= gyrothreshold else np.nan for i in range(len(gyroy))]
gyrozMovement = [10000 if gyroz2[i] >= gyrothreshold else np.nan for i in range(len(gyroz))]

plt.subplot(3,2,1)
line1, = plt.plot(time, accx2,label='Accelerometer X')
line2, = plt.plot(time, accxMovement,label='Movement')
plt.ylabel('Acceleration')
plt.title('Accelerometer X with movement')
plt.xlabel('Time (ms)')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(3,2,3)
line3, = plt.plot(time, accy2,label='Accelerometer Y')
line4, = plt.plot(time, accyMovement,label='Movement')
plt.ylabel('Acceleration')
plt.title('Accelerometer Y with movement')
plt.xlabel('Time (ms)')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(3,2,5)
line5, = plt.plot(time, accz2,label='Accelerometer Z')
line6, = plt.plot(time, acczMovement,label='Movement')
plt.ylabel('Acceleration')
plt.title('Accelerometer Z with movement')
plt.xlabel('Time (ms)')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(3,2,2)
line7, = plt.plot(time, gyrox2,label='Gyroscope X')
line8, = plt.plot(time, gyroxMovement,label='Movement')
plt.ylabel('Rotation')
plt.title('Gyroscope X with movement')
plt.xlabel('Time (ms)')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(3,2,4)
line9, = plt.plot(time, gyroy2,label='Gyroscope Y')
line10, = plt.plot(time, gyroyMovement,label='Movement')
plt.ylabel('Rotation')
plt.title('Gyroscope Y with movement')
plt.xlabel('Time (ms)')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(3,2,6)
line11, = plt.plot(time, gyroz2,label='Gyroscope Z')
line12, = plt.plot(time, gyrozMovement,label='Movement')
plt.ylabel('Rotation')
plt.title('Gyroscope Z with movement')
plt.xlabel('Time (ms)')
plt.legend(loc='best')
plt.grid(True)

plt.show()