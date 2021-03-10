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
           
def movement(movementData, moveValue):
    """
    movementData: List of when there is movement, binary list, either has np.nan or moveValue for a value
    moveValue: int, whatever value is set to denote motion
    
    This function goes through movementData and tracks how long the movement is and what index it starts at
    Returns: a list [noMoveLen, indexList]
    """
    moveLen = []
    indexList = []
    moveStartStop = []
    start = False
    moveCount = 0
    for i in range(len(movementData)):
        if(movementData[i]==moveValue and not start):
            start = True
            indexList.append(i)
            moveCount+=1
        elif(movementData[i]==moveValue and start):
            moveCount+=1
        elif(movementData[i]!=moveValue and start):
            moveLen.append(moveCount)
            moveCount = 0
            start = False
    if(moveCount!=0):
        moveLen.append(moveCount)
    for i in range(len(indexList)):
        moveStartStop.append([indexList[i],indexList[i]+moveLen[i]-1])
    return [moveLen, indexList, moveStartStop]
 
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

#Movement 
accMovement = [1 if accRMS[i] >= accLowerThreshold else np.nan for i in range(len(accRMS))]
gyroMovement = [1 if gyroRMS[i] >= gyroLowerThreshold else np.nan for i in range(len(gyroRMS))]

#Calculates the intermovement interval
accInterMov = movement(accMovement, 1)
gyroInterMov = movement(gyroMovement, 50)

kwargs = dict(alpha=0.75, bins=800)

plt.subplot(2,1,1)
plt.hist(accInterMov[0], **kwargs)
plt.ylabel('Frequency')
plt.xlabel('Accelerometer Intermovement Time')
plt.title('Accelerometer Histogram')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(2,1,2)
plt.hist(gyroInterMov[0], **kwargs)
plt.ylabel('Frequency')
plt.xlabel('Gyroscope Intermovement Time')
plt.title('Gyroscope Histogram')
plt.legend(loc='best')
plt.grid(True)

plt.show()