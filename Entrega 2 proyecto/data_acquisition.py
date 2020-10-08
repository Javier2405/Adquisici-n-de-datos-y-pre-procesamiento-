#------------------------------------------------------------------------------------------------------------------
#   Sample program for data acquisition and recording.
#------------------------------------------------------------------------------------------------------------------
import time
import socket
import numpy as np
import matplotlib.pyplot as plt
import threading as th
import random
from sklearn import svm
from sklearn.model_selection import KFold

# Data configuration
n_channels = 5
samp_rate = 256
emg_data = [[] for i in range(n_channels)]
print(emg_data)
samp_count = 0
UDP_IP = '127.0.0.1'
UDP_PORT = 8000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.01)



# Data acquisition

def getData():
    # Socket configuration
    global samp_count
    global n_channels
    start_time = time.time()
    while True:
        try:
            data, _ = sock.recvfrom(1024*1024)                        
            values = np.frombuffer(data)       
            ns = int(len(values)/n_channels)
            samp_count+=ns        

            for i in range(ns):
                for j in range(n_channels):
                    emg_data[j].append(values[n_channels*i + j])
                
            elapsed_time = time.time() - start_time
            if (elapsed_time > 1):
                start_time = time.time()
                '''print ("Muestras: ", ns)
                print ("Cuenta: ", samp_count)
                print ("Última lectura: ", [row[samp_count-1] for row in emg_data])
                print("")'''
        except socket.timeout:
            pass 
        #mpl.plt()

mythread=th.Thread(target= getData,daemon=True)

time.sleep(1)
mythread.start()
#t.join()
print("starting simulation")

# Training model

x = np.loadtxt("x_training.txt",dtype=float)
y = np.loadtxt("y_training.txt",dtype=float)

#print(x)
#print("-----------------------")
#print(y)

#SVM LINEAR
clf_linear = svm.SVC(kernel = 'linear')
clf_linear.fit(x, y)

# Live data predictions
past_samp=0
rangeinicial=0
rangefinal=samp_rate*5
fig, axs = plt.subplots(2, 2)
fig.tight_layout(pad=3.0)
while samp_count<samp_rate*5:
    time.sleep(0.5)
while True:
    power, freq = axs[1,0].psd(emg_data[0][samp_count-samp_rate:samp_count], NFFT = samp_rate, Fs = samp_rate)
    start_freq = next(x for x, val in enumerate(freq) if val >= 4.0)
    end_freq = next(x for x, val in enumerate(freq) if val >= 60.0)
    start_index = np.where(freq >= start_freq)[0][0]
    end_index = np.where(freq >= end_freq)[0][0]
    power = power[start_index:(end_index+1)]

    power2, freq2 = axs[1,0].psd(emg_data[2][samp_count-samp_rate:samp_count], NFFT = samp_rate, Fs = samp_rate)
    start_freq2 = next(x for x, val in enumerate(freq2) if val >= 4.0)
    end_freq2 = next(x for x, val in enumerate(freq2) if val >= 60.0)
    start_index2 = np.where(freq2 >= start_freq2)[0][0]
    end_index2 = np.where(freq2 >= end_freq2)[0][0]
    power2 = power2[start_index:(end_index+1)]
    
    power_total = np.concatenate((power,power2),axis=0)

    pred = clf_linear.predict([power_total])
    print(pred)
    
    difference=samp_count-past_samp
    past_samp=samp_count
    
            
     
    
    

