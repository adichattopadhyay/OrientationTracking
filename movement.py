import pandas as pd
import numpy
import math
import matplotlib.pyplot as plt
import scipy
import sys  
from scipy import signal
from scipy import pi
import numpy as np    
from scipy.signal import butter, lfilter, freqz

import os

def combineMovement(accMov, gyroMov, moveValue):
    print(len(accMov))
    print(len(gyroMov))
    movList = []
    for i in range(len(accMov)):
        if(accMov[i] == moveValue or gyroMov[i] == moveValue):
            if(gyroMov[i] != moveValue):
                movList.append(accMov[i])
            else:
                movList.append(gyroMov[i])
        else:
            movList.append(np.nan)
    return movList                
            
def movement(movementData, moveValue):
    """
    movementData: List of when there is movement, binary list, either has np.nan or moveValue for a value
    moveValue: int, whatever value is set to denote motion
    This function goes through movementData and tracks how long the movement is and what index it starts at
    Returns a list [noMoveLen, indexList]
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
    
def noMovement(movementData, moveValue):
    """
    movementData: List of when there is movement, binary list, either has np.nan or moveValue for a value
    moveValue: int, whatever value is set to denote motion
    This function goes through movementData and tracks how long the gap between the movements is and what index it starts at
    Returns a list [noMoveLen, indexList]
    """
    noMoveLen = []
    indexList = []
    noMoveStartStop = []
    start = False
    moveCount = 0
    for i in range(len(movementData)):
        if(movementData[i]!=moveValue and not start):
            start = True
            indexList.append(i)
            moveCount+=1
        elif(movementData[i]!=moveValue and start):
            moveCount+=1
        elif(movementData[i]==moveValue and start):
            noMoveLen.append(moveCount)
            moveCount = 0
            start = False  
    if(moveCount!=0):
        noMoveLen.append(moveCount)
    for i in range(len(indexList)):
        noMoveStartStop.append([indexList[i],indexList[i]+noMoveLen[i]-1])
    #print(*noMoveLen) Need to multiply by 40 to get the actual time
    #print(*indexList)
    return [noMoveLen, indexList, noMoveStartStop]

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
    rms = []
    for i in range(len(lists[0])):
        rms.append(math.sqrt((1/3)*((lists[0][i]**2)+(lists[1][i]**2)+(lists[2][i]**2))))
    return rms

fileName = input("Which file do you want?: ")

df = pd.read_csv(os.getcwd()+"/Data/"+fileName+".csv")

time = df['time'].to_list()
accx = [x / 4096 for x in df['accelerometerX'].to_list()]
accy = [x / 4096 for x in df['accelerometerY'].to_list()]
accz = [x / 4096 for x in df['accelerometerZ'].to_list()]
gyrox = [x / 131 for x in df['gyroscopeX'].to_list()]
gyroy = [x / 131 for x in df['gyroscopeY'].to_list()]
gyroz = [x / 131 for x in df['gyroscopeZ'].to_list()]

accx2 = butter_highpass_filter(accx, 0.2, 25)
accy2 = butter_highpass_filter(accy, 0.2, 25)
accz2 = butter_highpass_filter(accz, 0.2, 25) 

accRMS = rms3([accx2, accy2, accz2])

gyrox2 = butter_highpass_filter(gyrox, 0.05, 25)
gyroy2 = butter_highpass_filter(gyroy, 0.05, 25)
gyroz2 = butter_highpass_filter(gyroz, 0.05, 25) 

gyroRMS = rms3([gyrox2, gyroy2, gyroz2])

accLowerThreshold = 0.1
accUpperThreshold = 8000
gyroLowerThreshold = 10
gyroUpperThreshold = 8000

#accMovement = [10000 if accRMS[i] >= accLowerThreshold and accRMS[i] <= accUpperThreshold else np.nan for i in range(len(accRMS))]
#gyroMovement = [10000 if gyroRMS[i] >= gyroLowerThreshold and gyroRMS[i] <= gyroUpperThreshold else np.nan for i in range(len(gyroRMS))]

accMovement = [1 if accRMS[i] >= accLowerThreshold else np.nan for i in range(len(accRMS))]
gyroMovement = [1 if gyroRMS[i] >= gyroLowerThreshold else np.nan for i in range(len(gyroRMS))]

accMov = movement(accMovement, 1)
gyroMov = movement(gyroMovement, 1)

accInterMov = noMovement(accMovement, 1)
gyroInterMov = noMovement(gyroMovement, 1)

movList = combineMovement(accMovement, gyroMovement, 1)
mov = movement(movList, 1)
movInter = noMovement(movList, 1)

print(movInter[2][0:10])
print(mov[2][0:10])

plt.subplot(3,1,1)
plt.tight_layout(h_pad=1)
line1, = plt.plot(time, accRMS,label='Accelerometer')
line2, = plt.plot(time, accMovement,label='Movement')
plt.ylabel('Acceleration (*insert units*)')
plt.title('Accelerometer data over time')
plt.xlabel('Time (ms)')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(3,1,2)
line3, = plt.plot(time, gyroRMS,label='Gyroscope')
line4, = plt.plot(time, gyroMovement,label='Movement')
plt.ylabel('Gyroscope (*insert units*)')
plt.xlabel('Time (ms)')
plt.title('Gyroscope data over time')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(3,1,3)
line5, = plt.plot(time, movList, label='Movement')
plt.ylabel('Movement')
plt.xlabel('Time (ms)')
plt.title('Movement over time')
plt.legend(loc='best')
plt.grid(True)

plt.show()

"""
Put the code on github (Nilanjan's github: nilanbumbc)
1. APPLY THE THRESHOLD VALUE OF 10 TO COMBINE MOVEMENTS
2. Do the orientation for the no movement
_________________________________________________
1. Fill the gaps with the threshold method (done)
   To find the threshold:
        Look at the **intramovement** intervals
        Plot a distribution curve of the intramovement intervals 
        (List of the distances in the gap)
2. Combine the accel and gyro MOVEMENT
    Easiest way to do that is take a union of the two (not addition the U thing) 
    More like an or
3. Isolate periods of time when there is movement vs no movement (done)
    Different orientation algorithm for each
    No movement:
        Using the gravity component from the accelerometer (go back to x,y,z component)
        Measuring orientation on yaw, pitch, and roll
        Find 3 angles for each sample (trig, tan I think) 
        See what is necessary to find the overall orientation (ex. averages)
        Orientation is not very stable
        Unsupervised ML 
    Movement:
_________________________________________________________________________

5 more csvs
Normalization factors (changing the units to g)
Two threshold method (Lower [above the noise floor] and upper)
Combine accelerometer and gyroscope x y z with root mean square
Fix the threshold for gyroscope     
"""