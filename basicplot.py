import pandas as pd
import numpy
import matplotlib.pyplot as plt
from scipy import signal
from scipy import pi
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

plt.subplot(2, 1, 1)
line1, = plt.plot(time, accx,label='Accelerometer X')
line2, = plt.plot(time, accy,label='Accelerometer Y')
line3, = plt.plot(time, accz,label='Accelerometer Z')

plt.ylabel('Acceleration')
plt.title('Accelerometer data over time')
plt.xlabel('Time (ms)')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(2, 1, 2)
line4, = plt.plot(time, gyrox,label='Gyroscope X')
line5, = plt.plot(time, gyroy,label='Gyroscope Y')
line6, = plt.plot(time, gyroz,label='Gyroscope Z')
plt.ylabel('Gyroscope')
plt.xlabel('Time (ms)')
plt.title('Gyroscope data over time')
plt.legend(loc='best')
plt.grid(True)

plt.show()
