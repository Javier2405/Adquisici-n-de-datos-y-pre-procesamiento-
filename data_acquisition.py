#------------------------------------------------------------------------------------------------------------------
#   Sample program for data acquisition and recording.
#------------------------------------------------------------------------------------------------------------------
import time
import socket
import numpy as np
import matplotlib.pyplot as plt
import threading as th
import random

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
                print ("Muestras: ", ns)
                print ("Cuenta: ", samp_count)
                print ("Última lectura: ", [row[samp_count-1] for row in emg_data])
                print("")
        except socket.timeout:
            pass 
        #mpl.plt()

mythread=th.Thread(target= getData,daemon=True)

time.sleep(1)
mythread.start()
#t.join()
print("starting plotting")

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

    power2, freq2 = axs[1,0].psd(emg_data[2][samp_count-samp_rate:samp_count], NFFT = samp_rate, Fs = samp_rate)
    start_freq2 = next(x for x, val in enumerate(freq2) if val >= 4.0)
    end_freq2 = next(x for x, val in enumerate(freq2) if val >= 60.0)
    start_index2 = np.where(freq2 >= start_freq2)[0][0]
    end_index2 = np.where(freq2 >= end_freq2)[0][0]

    
    difference=samp_count-past_samp
    past_samp=samp_count
    xlist=[]
    for x in range(rangeinicial,rangefinal):
        xlist.append(x/samp_rate)
    print(rangeinicial,rangefinal)
    for ax in axs.flat:
        ax.cla()
        #ax.set_ylim(-100,100)
    axs[0,0].set_title("Señales EMG del canal 1")
    axs[0,0].set(xlabel='Tiempo (s)', ylabel='micro V')
    axs[0,1].set_title("Señales EMG del canal 2")
    axs[0,1].set(xlabel='Tiempo (s)', ylabel='micro V')
    axs[1,0].set_title("PSD calculado del canal 1")
    axs[1,0].set(xlabel='Frecuencia (Hz)', ylabel='Power')
    axs[1,1].set_title("PSD calculado del canal 2")
    axs[1,1].set(xlabel='Frecuencia (Hz)', ylabel='Power')

    axs[0,0].plot(xlist, emg_data[0][samp_count-samp_rate*5:samp_count])
    axs[0,1].plot(xlist, emg_data[2][samp_count-samp_rate*5:samp_count])
    axs[1,0].plot(freq[start_index:end_index], power[start_index:end_index])
    axs[1,1].plot(freq2[start_index2:end_index2],power2[start_index2:end_index2])
    for i in range(difference):
        rangefinal+=1
        rangeinicial+=1
    past_count=samp_count
    plt.pause(0.05)


plt.show()
            
     
    
    

