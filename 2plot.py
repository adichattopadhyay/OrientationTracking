import pandas as pd
import math
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

def rms3(lists):
    """
    lists: List of 3 lists all of the same length
    
    This function goes through the lists and for each point calculates the Root Mean Squared
    Returns: A list of the values root mean squared. 
    """
    rms = []
    for i in range(len(lists[0])):
        rms.append(math.sqrt((1/3)*((lists[0][i]**2)+(lists[1][i]**2)+(lists[2][i]**2))))
    return rms

fileName = input("Which file do you want?: ")

df = pd.read_csv(os.getcwd()+"/Data/"+fileName+".csv")

time = df['time'].to_list()

#Normalization factors
accx = [x / 4096 for x in df['accelerometerX'].to_list()]
accy = [x / 4096 for x in df['accelerometerY'].to_list()]
accz = [x / 4096 for x in df['accelerometerZ'].to_list()]
gyrox = [x / 131 for x in df['gyroscopeX'].to_list()]
gyroy = [x / 131 for x in df['gyroscopeY'].to_list()]
gyroz = [x / 131 for x in df['gyroscopeZ'].to_list()]

#Centers the data at 0
accx2 = butter_highpass_filter(accx, 0.2, 25)
accy2 = butter_highpass_filter(accy, 0.2, 25)
accz2 = butter_highpass_filter(accz, 0.2, 25) 
gyrox2 = butter_highpass_filter(gyrox, 0.05, 25)
gyroy2 = butter_highpass_filter(gyroy, 0.05, 25)
gyroz2 = butter_highpass_filter(gyroz, 0.05, 25) 

#Calculates the RMS of the data
accRMS = rms3([accx2, accy2, accz2])
gyroRMS = rms3([gyrox2, gyroy2, gyroz2])

#Thresholds of the data
accLowerThreshold = 0.1
gyroLowerThreshold = 10

#Movement algorithm
accMovement = [1 if accRMS[i] >= accLowerThreshold else np.nan for i in range(len(accRMS))]
gyroMovement = [50 if gyroRMS[i] >= gyroLowerThreshold else np.nan for i in range(len(gyroRMS))]

plt.subplot(2,1,1)
line1, = plt.plot(time, accRMS,label='Accelerometer')
line2, = plt.plot(time, accMovement,label='Movement')
plt.ylabel('Acceleration (g)')
plt.title('Accelerometer data over time')
plt.xlabel('Time (ms)')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(2,1,2)
line3, = plt.plot(time, gyroRMS,label='Gyroscope')
line4, = plt.plot(time, gyroMovement,label='Movement')
plt.ylabel('Gyroscope (dps)')
plt.xlabel('Time (ms)')
plt.title('Gyroscope data over time')
plt.legend(loc='best')
plt.grid(True)

plt.show()