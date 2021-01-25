import pandas as pd
import numpy
import matplotlib.pyplot as plt

df = pd.read_csv("imu.csv")
 
time = df['time'].to_list()
accx = df['accelerometerX'].to_list()
accy = df['accelerometerZ'].to_list()
accz = df['accelerometerY'].to_list()
gyrox = df['gyroscopeX'].to_list()
gyroy = df['gyroscopeY'].to_list()
gyroz = df['gyroscopeZ'].to_list()

plt.subplot(2, 1, 1)
line1, = plt.plot(time, accx,label='Accelerometer X')
line2, = plt.plot(time, accy,label='Accelerometer Y')
line3, = plt.plot(time, accz,label='Accelerometer Z')

plt.ylabel('Acceleration (*insert units*)')
plt.title('Accelerometer data over time')
plt.xlabel('Time (ms)')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(2, 1, 2)
line4, = plt.plot(time, gyrox,label='Gyroscope X')
line5, = plt.plot(time, gyroy,label='Gyroscope Y')
line6, = plt.plot(time, gyroz,label='Gyroscope Z')
plt.ylabel('Gyroscope (*insert units*)')
plt.xlabel('Time (ms)')
plt.title('Gyroscope data over time over Time')
plt.legend(loc='best')
plt.grid(True)

plt.show()
