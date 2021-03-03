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
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import os

def divideChunk(chunkSize, overlap, data):
    nextSample = round(chunkSize*overlap)
    indexList = []
    for i in range(0, len(data), nextSample):
        if(i+chunkSize-1 >= len(data)):
            indexList.append([i, len(data)-1])
        else:
            indexList.append([i,i+chunkSize-1])
    print("\nFirst 10 for index list:", indexList[0:10])
    return indexList

def nightAvg(windowSize, overlap, data, movement):
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
    accx = np.unwrap(acc[0])
    accy = np.unwrap(acc[1])
    accz = np.unwrap(acc[2])
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
    count = 0
    indexList = divideChunk(windowSize, overlap, data)
    avgData = []
    print("data len", len(data))
    print("index len", len(indexList))
    for i in range(len(data)):
        try:
            if(indexList[count][0]==i):
                avg = sum(data[indexList[count][0]:indexList[count][1]])/windowSize  
                avgData.append(avg)
                if(count!=len(indexList)-1):
                    count+=1
            else:
                avgData.append(avgData[i-1])
        except:
            print("count", count)
            print("i", i)
            print("len avgData", len(avgData))
            print("Last 10 for index list: ", indexList[-10:])
            exit()
    return avgData

def expandList(arr):
    arrExpand = []
    for i in range(len(arr)):
        for j in arr[i]:
            arrExpand.append(j)
    return arrExpand

def noMovOrientation(acc, interMovement):
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
    return [[yaw, pitch, roll], orientation, orientationAvg, time2]

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
    #print(len(accMov))
    #print(len(gyroMov))
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
movFinal = movement(movListFinal, 1)
movInterFinal = noMovement(movListFinal, 1)

#Calculate Orientation [[yaw, pitch, roll], orientation, orientationAvg, time]
orientation = noMovOrientation([accx2, accy2, accz2], movInterFinal[2])

#Extract from orientation
yaw = orientation[0][0]
pitch = orientation[0][1]
roll = orientation[0][2]
interMovOrientation = orientation[1]
interMovOrientationAvg = orientation[2]
interMovTime = orientation[3]

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

print("\n------------------------Orientation------------------------\n")
print("The following are for the first 10 orientations for the first intermovement interval")
print("Yaw:")
print(yaw[0][:10])
print("Pitch:")
print(pitch[0][:10])
print("Roll:")
print(roll[0][:10])

print("\nOrientation:")
print(interMovOrientation[0][:10])
print("\nThe following is the first 10 average orientations")
print(interMovOrientationAvg[:10])

print("\n------------------------Time------------------------\n")
print("Time:")
print(interMovTime[0:20])
print(interMovTime[72:92])

#Graph the roll and pitch intermovement interval
#If they look consistent then there is no preprocessing step needed
#Across the five nights
#Forget about yaw for now

#print(len(pitch))
#print(pitch[1540:1570])            

#Expands the roll, pitch, and yaw lists
#yawExpand = expandList(yaw)


pitchExpand = expandList(pitch)
rollExpand = expandList(roll)
pitchAvg = windowAvg(100, 9/10, pitchExpand)
rollAvg = windowAvg(100, 9/10, rollExpand)
pitchFinal = []
rollFinal = []
cntr = 0


wait=False

for i in range(len(time)):
    if(i>len(interMovTime)-1):
        pitchFinal.append(np.nan)
        rollFinal.append(np.nan)
    elif(time[i]==interMovTime[i-cntr]):
        pitchFinal.append(pitchAvg[i-cntr])
        rollFinal.append(rollAvg[i-cntr])
    elif(time[i]!=interMovTime[i-cntr]):
        pitchFinal.append(np.nan)
        rollFinal.append(np.nan)
        cntr+=1

orientation2 = orient([accx2, accy2, accz2])
#s = int(input("How many seconds do you want the window to be: "))
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

df1 = pd.DataFrame(
    {
        'pitch':nightAverage[0],
        'roll':nightAverage[1]
    }
)

nC = int(input("How many clusters?: "))

kmeans = KMeans(n_clusters=nC).fit(df1)
centroids = kmeans.cluster_centers_
print(centroids)

colors = ['red', 'green']
fig = plt.figure(figsize=(8,8))
#plt.scatter(nightAverage[0], nightAverage[1], c=nightAverage[2], cmap=matplotlib.colors.ListedColormap(colors))
plt.scatter(nightAverage[0], nightAverage[1],  c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.title('Pitch vs Roll (' + str(s) + " seconds)")
plt.xlabel("Pitch")
plt.ylabel("Roll")
plt.grid(True)

plt.show()

"""
Do phase unwrapping before clustering https://scikit-image.org/docs/dev/auto_examples/filters/plot_phase_unwrap.html
Overlay overnight data with the cluster
    Name the clusters 1, 2, 3, 4, ...
    Each cluster appears at a certain time, at that time, the value of the graph I'm plotting coresponds to a cluster
    So per window overlay it with the id
    Show the accRMS
    Look at the example PNG

"""

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