import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import scipy
import sys  
from scipy import signal
from scipy import pi
import numpy as np    
from scipy.signal import butter, lfilter, freqz

import os

def divideChunk(chunkSize, overlap, data):
    """
    chunkSize: The size of each chunk that the data is going to divided into
    overlap: Fraction. How much overlap is in each chunk, if say you want 90% overlap, you want to input 1/10 due to how overlap is implemented. 
    data: The data that will be divided
    
    This function takes data, divides that data into chunks of size chunkSize with an overlap of overlap 
    Returns: A list that contains the start and end index of every chunk.
    """
    nextSample = round(chunkSize*overlap)
    indexList = []
    for i in range(0, len(data), nextSample):
        if(i+chunkSize-1 >= len(data)):
            indexList.append([i, len(data)-1])
        else:
            indexList.append([i,i+chunkSize-1])
    return indexList

def nightAvg(windowSize, overlap, data, movement):
    """
    windowSize: The size of each window that the data is going to divided into
    overlap: Fraction. How much overlap is in each window, if say you want 90% overlap, you want to input 1/10 due to how overlap is implemented. 
    data: The data that will be divided and used to calculate the average pitch and roll for each chunk. This data is meant to be orientation data.
    movement: A list of the movements
    
    This function takes the orientation data and divides it into chunks using the divideChunk() function. 
    Each chunk is associated with an average pitch and roll which is just the averages of the pitches and rolls in the chunk.
    Finally, each chunk is classified as a movement chunk if >= 50% of the chunk is movement. If not it is a non-movement chunk.
    Returns: The average Pitch, average Roll, and movement for every chunk.
    """
    count = 0
    movCount = 0
    noMovCount = 0
    indexList = divideChunk(windowSize, overlap, data[0])
    pitch = data[0]
    roll = data[1]
    moveList = movement
    avgPitch = []
    avgRoll = []
    movementList = []
    for i in range(len(pitch)):        
        if(indexList[count][0]==i):
            avgP = sum(pitch[indexList[count][0]:indexList[count][1]])/windowSize 
            avgR = sum(roll[indexList[count][0]:indexList[count][1]])/windowSize   
            avgPitch.append(avgP)
            avgRoll.append(avgR)             
        elif(indexList[count][1]==i):
            if(movCount >= noMovCount):
                movementList.append(True)
            elif(movCount < noMovCount):   
                movementList.append(False)
            movCount = 0
            noMovCount = 0
            count+=1            
        if(moveList[i]!=1):
            noMovCount+=1
        elif(moveList[i]==1):
            movCount+=1
    return [avgPitch, avgRoll, movementList]
    
def orient(acc):
    """
    acc: Accelerometer data
    
    This function is used to calculate the pitch and roll for every data point in the accelerometer data.
    Returns: The pitch, roll, and orientation for every data point.
    """
    
    #Phase unwrapping
    accx = acc[0]
    accy = acc[1]
    accz = acc[2]
    pitch = []
    roll = []
    orientation = []
    for i in range(len(accx)):
        rollVal = (math.atan2(accy[i],accz[i]) * 180/math.pi)
        pitchVal = math.atan2(accx[i], math.sqrt(accy[i]**2+accz[i]**2)) * 180/math.pi
        roll.append(rollVal)
        pitch.append(pitchVal)
        orientation.append((roll[i]+pitch[i])/2)
    return [pitch, roll, orientation]

def windowAvg(windowSize, overlap, data):
    """
    windowSize: The size of each window that the data is going to divided into
    overlap: Fraction. How much overlap is in each window, if say you want 90% overlap, you want to input 1/10 due to how overlap is implemented. 
    data: The data that will be divided and used to calculate the moving average with
    
    This function takes any list of data points and calculates the moving average with a window size windowSize and overlap overLap.
    Returns: The result of the moving average filter.
    """
    count = 0
    indexList = divideChunk(windowSize, overlap, data)
    avgData = []
    for i in range(len(data)):
        if(indexList[count][0]==i):
            avg = sum(data[indexList[count][0]:indexList[count][1]])/windowSize  
            avgData.append(avg)
            if(count!=len(indexList)-1):
                count+=1
        else:
            avgData.append(avgData[i-1])
    return avgData

def expandList(arr):
    """
    arr: A double array, e.x. [[1,2][3,4]]
    
    This function takes a double array and makes it into a singlular array
    Returns: A singular array. 
    """
    arrExpand = []
    for i in range(len(arr)):
        for j in arr[i]:
            arrExpand.append(j)
    return arrExpand

def noMovOrientation(acc, interMovement):
    """
    acc: Accelerometer data
    interMovement: Double array containing the start and end indexes of the intermovement intervals
    
    This function goes through the intermovement periods and calculates the roll and pitch for each data point. 
    For each specific array that contains a start and end index, it calculates and overall pitch and overall orientation. 
    Returns: A list of the individual yaw, pitch, and roll, individal orientation, the orientation averages, and a list of the times of each of these points.
    """
    accx = acc[0]
    accy = acc[1]
    accz = acc[2]
    yaw = []
    pitch = []
    roll = []
    orientation = []
    orientationAvg = []
    time2 = []
    for i in range(len(interMovement)):
        yaw.append([])
        pitch.append([])
        roll.append([])
        orientation.append([])
    print(len(interMovement))
    for i in range(len(interMovement)):
        for j in range(interMovement[i][0], interMovement[i][1]+1):
            time2.append(time[j])
            rollVal = (math.atan2(accy[j],accz[j]) * 180/math.pi)
            pitchVal = math.atan2(accx[j], math.sqrt(accy[j]**2+accz[j]**2)) * 180/math.pi
            roll[i].append(rollVal)
            pitch[i].append(pitchVal)
            #From https://samselectronicsprojects.blogspot.com/2014/07/getting-roll-pitch-and-yaw-from-mpu-6050.html 
    for i in range(len(roll)):
        for j in range(len(roll[i])):
            orientation[i].append((roll[i][j]+pitch[i][j])/2)
    for i in range(len(orientation)):
        orientationAvg.append(sum(orientation[i])/len(orientation[i]))
    return [[yaw, pitch, roll], orientation, orientationAvg, time2]

def finalMovement(movement, threshold, moveValue):
    """
    movement: Movement data
    threshold: The maximum gap length for the gap to be considered movement
    movValue: The value used to designate movement in the list movement
    
    This function goes through the movementList and fills up all the gaps that are <= threshold.
    Returns: A list with gaps filled in the movement
    """
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
                started=False
            else:
                count+=1
        elif(movement[i]==moveValue and started):
            for j in range(count+1):
                finalMovement.append(moveValue)
            count = 0
            started = False
    return finalMovement

def combineMovement(accMov, gyroMov, moveValue):
    """
    accMov: The movement calculated with the accelerometer
    gyroMov: The movement calculated with the gyroscrope
    movValue: The value used to designate movement in the lists accMov and gyroMov
    
    This function goes through both the accelerometer movement and gyroscope movement and combines the two.
    Returns: A list with the combined accelerometer and gyroscope movement.  
    """
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
    
def noMovement(movementData, moveValue):
    """
    movementData: List of when there is movement, binary list, either has np.nan or moveValue for a value
    moveValue: int, whatever value is set to denote motion
    
    This function goes through movementData and tracks how long the gap between the movements is and what index it starts at
    Returns: A list [noMoveLen, indexList]
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

#Movement with gaps of < gapLen accounted for
gapLen = 25
accMovementFinal = finalMovement(accMovement, gapLen, 1)
gyroMovementFinal = finalMovement(gyroMovement, gapLen, 1)

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

#Second orientation calculation, but for the whole data
orientation2 = orient([accx2, accy2, accz2])

#Divides the orientation into chunks
s = 60
nightAverage = nightAvg(s*45, 1, orientation2, movListFinal)

print("\n------------------------Night Average------------------------\n")
print("Average Pitch: ")
print(nightAverage[0][0:10])
print("Average Roll: ")
print(nightAverage[1][0:10])
print("Is Moving: ")
print(nightAverage[2][0:10])

print("len of pitch", len(nightAverage[0]))
print("len of roll", len(nightAverage[1]))
print("len of moving", len(nightAverage[2]))
print("\n")


colors = ['red', 'green']
fig = plt.figure(figsize=(8,8))
plt.scatter(nightAverage[0], nightAverage[1], c=nightAverage[2], cmap=matplotlib.colors.ListedColormap(colors))
plt.title('Pitch vs Roll (' + str(s) + " seconds)")
plt.xlabel("Pitch")
plt.ylabel("Roll")
plt.grid(True)

plt.show()

"""
plt.subplot(2,1,1)
line1, = plt.plot(time, pitchFinal, label='pitch')
line2, = plt.plot(time, movListFinal, label='Movement Final')
plt.ylabel('Pitch (*insert units*)')
plt.title('Pitch')
plt.xlabel('Time (ms)')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(2,1,2)
line3, = plt.plot(time, rollFinal, label='roll')
line4, = plt.plot(time, movListFinal, label='Movement Final')
plt.ylabel('Roll (*insert units*)')
plt.title('Roll')
plt.xlabel('Time (ms)')
plt.legend(loc='best')
plt.grid(True)

plt.show()
"""
"""
Take the whole night of data
Divide it into 30 second windows (30*25=750 samples)
Keep an overlap factor like with the other one. For the timebeing have 0 overlap
For each window:
    1. Average Pitch
    2. Average Roll
    3. Moving vs Not
        If > 50% moving, then person is moving
        If < 50% moving, then person is not moving
Plot a scatter plot of the pitch (x) vs the roll (y)
Color of dots:
    1. Moving
    2. Not moving
"""