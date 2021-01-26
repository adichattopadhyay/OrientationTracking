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

def noMovOrientation(acc, interMovement):
    accx = acc[0]
    accy = acc[1]
    accz = acc[2]
    yaw = []
    pitch = []
    roll = []
    orientation = []
    orientationAvg = []
    for i in range(len(interMovement)):
        yaw.append([])
        pitch.append([])
        roll.append([])
        orientation.append([])
    
    for i in range(len(interMovement)):
        for j in range(interMovement[i][0], interMovement[i][1]+1):
            roll[i].append(math.atan2(accy[j],accz[j]) * 180/math.pi)
            pitch[i].append(math.atan2(accx[j], math.sqrt(accy[j]**2+accz[j]**2) * 180/math.pi))
            #yaw[i].append()
            #Roll = atan2(Y, Z) * 180/PI;
            #Pitch = atan2(X, sqrt(Y*Y + Z*Z)) * 180/PI;
            #From https://samselectronicsprojects.blogspot.com/2014/07/getting-roll-pitch-and-yaw-from-mpu-6050.html 
    for i in range(len(roll)):
        for j in range(len(roll[i])):
            #orientation[i].append((roll[i][j]+pitch[i][j]+yaw[i][j])/3) For when yaw is implemented
            orientation[i].append((roll[i][j]+pitch[i][j])/2)
    for i in range(len(orientation)):
        orientationAvg.append(sum(orientation[i])/len(orientation[i]))
    return [[yaw, pitch, roll], orientation, orientationAvg]

def finalMovement(movement, threshold, moveValue):
    started = False
    count = 0
    finalMovement = []
    for i in range(len(movement)):
        if((movement[i]!=moveValue and not started) or (movement[i]==moveValue and started)):
            finalMovement.append(movement[i])
        elif(movement[i]==moveValue and not started):
            started = True
            finalMovement.append(movement[i])
        elif(movement[i]!=moveValue and started):
            if(count == threshold):
                for j in range(count+1):
                    finalMovement.append(np.nan)
                count = 0
                #print("I'm in the third elif")
                #print(len(finalMovement))
                #print(i)
                started=False
            else:
                count+=1
        elif(movement[i]==moveValue and started):
            for j in range(count+1):
                finalMovement.append(moveValue)
            count = 0
            started = False
            #print("I'm in the fourth elif---------------------")
            #print(len(finalMovement))
            #print(i)
    #print("All done")
    #print(len(finalMovement))
    #print(len(movement))
    return finalMovement

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

#Movement 
accMovement = [1 if accRMS[i] >= accLowerThreshold else np.nan for i in range(len(accRMS))]
gyroMovement = [1 if gyroRMS[i] >= gyroLowerThreshold else np.nan for i in range(len(gyroRMS))]

#Movement with gaps of < 10 accounted for
accMovementFinal = finalMovement(accMovement, 10, 1)
gyroMovementFinal = finalMovement(gyroMovement, 10, 1)

#Isolation of the movement intervals (with gaps)
accMov = movement(accMovement, 1)
gyroMov = movement(gyroMovement, 1)

#Isolation of the movement intervals (without gaps)
accMovFinal = movement(accMovementFinal, 1)
gyroMovFinal = movement(gyroMovementFinal, 1)

#Isolation of the intermovement intervals (with gaps)
accInterMov = noMovement(accMovement, 1)
gyroInterMov = noMovement(gyroMovement, 1)

#Isolation of the intermovement intervals (without gaps)
accInterMovFinal = noMovement(accMovementFinal, 1)
gyroInterMovFinal = noMovement(gyroMovementFinal, 1)

#Combined movement (with gaps)
movList = combineMovement(accMovement, gyroMovement, 1)
mov = movement(movList, 1)
movInter = noMovement(movList, 1)

#Combined movement (without gaps)
movListFinal = combineMovement(accMovementFinal, gyroMovementFinal, 1)
movFinal = movement(movListFinal, 1)
movInterFinal = noMovement(movListFinal, 1)

print("\n------------------------With gaps------------------------\n")
print("Intermovement: ")
print(movInter[2][0:10])
print("Movement: ")
print(mov[2][0:10])

print("\n------------------------Without gaps------------------------\n")
print("Intermovement: ")
print(movInterFinal[2][0:10])
print("Movement:")
print(movFinal[2][0:10])

orientation = noMovOrientation([accx2, accy2, accz2], movInterFinal[2])
print("\n------------------------Orientation------------------------\n")
print("The following are for the first 10 orientations for the first intermovement interval")
print("Yaw:")
print(orientation[0][0][0][:10])
print("Pitch:")
print(orientation[0][1][0][:10])
print("Roll:")
print(orientation[0][2][0][:10])

print("\nOrientation:")
print(orientation[1][0][:10])
print("\nThe following is the first 10 average orientations")
print(orientation[2][:10])

pitch = []
roll = []
for i in range(len(orientation[0][1])):
    for j in range(len(orientation[0][1][i])):
        roll.append(orientation[0][2][i])
        pitch.append(orientation[0][1][i])

#Graph the roll and pitch intermovement interval
#If they look consistent then there is no preprocessing step needed
#Across the five nights
#Forget about yaw for now

plt.subplot(2,1,1)
line1, = plt.plot(time, pitch ,label='pitch')
plt.ylabel('Acceleration (*insert units*)')
plt.title('Pitch')
plt.xlabel('Time (ms)')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(2,1,2)
line1, = plt.plot(time, roll ,label='roll')
plt.ylabel('Acceleration (*insert units*)')
plt.title('Roll')
plt.xlabel('Time (ms)')
plt.legend(loc='best')
plt.grid(True)

plt.show()
"""
plt.subplot(3,1,1)
plt.tight_layout(h_pad=1)
line1, = plt.plot(time, accRMS,label='Accelerometer')
line2, = plt.plot(time, accMovementFinal,label='Movement Final')
plt.ylabel('Acceleration (*insert units*)')
plt.title('Accelerometer data over time')
plt.xlabel('Time (ms)')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(3,1,2)
line3, = plt.plot(time, gyroRMS,label='Gyroscope')
line4, = plt.plot(time, gyroMovementFinal,label='Movement Final')
plt.ylabel('Gyroscope (*insert units*)')
plt.xlabel('Time (ms)')
plt.title('Gyroscope data over time')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(3,1,3)
line5, = plt.plot(time, movListFinal, label='Movement Final')
line6, = plt.plot(time, movList, label = 'Movement (not Final)')
plt.ylabel('Movement')
plt.xlabel('Time (ms)')
plt.title('Movement over time')
plt.legend(loc='best')
plt.grid(True)

plt.show()
"""
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