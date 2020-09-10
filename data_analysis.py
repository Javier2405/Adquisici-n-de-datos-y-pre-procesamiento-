#------------------------------------------------------------------------------------------------------------------
#   Sample program for EMG data loading and manipulation.
#------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Read data file
#data = np.loadtxt("Abierto, cerrado, descanso.txt") 
#data = np.loadtxt("Izquierda, derecha, cerrado.txt")
data = np.loadtxt("Abierto, cerrado.txt")

samp_rate = 256
samps = data.shape[0]
n_channels = data.shape[1]

print('Número de muestras: ', data.shape[0])
print('Número de canales: ', data.shape[1])
print('Duración del registro: ', samps / samp_rate, 'segundos')
print(data)

# Time channel
time = data[:, 0]

# Data channels
chann1 = data[:, 1]
chann2 = data[:, 3]

posturas = []
channels = [chann1,chann2]

# Mark data
mark = data[:, 6]

training_samples = {}
for i in range(0, samps): 
    if mark[i] > 0: 
        print("Marca", mark[i], 'Muestra', i, 'Tiempo', time[i]) 

        if  (mark[i] > 100) and (mark[i] < 200):
            if(mark[i] not in posturas):
                posturas.append(mark[i])
            iniSamp = i
            condition_id = mark[i]
        elif mark[i] == 200:
            if not condition_id in training_samples.keys():
                training_samples[condition_id] = []
            training_samples[int(condition_id)].append([iniSamp, i])

print('Rango de muestras con datos de entrenamiento:', training_samples)
posturas.sort()

def graph15secs(postura, ind1, posx, posy):
    start_samp = training_samples[postura][ind1][0]
    #end_samp = training_samples[postura][ind1][1]
    win_size=256
    end_samp=start_samp+win_size

    axs_chann1_15secs[posx,posy].plot(time[start_samp:end_samp], chann1[start_samp:end_samp], color='green')
    axs_chann2_15secs[posx,posy].plot(time[start_samp:end_samp], chann2[start_samp:end_samp], color = 'red')
    axs_chann1_15secs[posx,posy].set_title(str(postura) + " "+str(ind1)+" , "+str(0))
    axs_chann2_15secs[posx,posy].set_title(str(postura) + " "+str(ind1)+" , "+str(0))
    axs_chann1_15secs[posx,posy].set(xlabel='Tiempo (s)', ylabel='micro V')
    axs_chann2_15secs[posx,posy].set(xlabel='Tiempo (s)', ylabel='micro V')
    
def psd15secs(postura, ind1, posx,posy):
    start_samp = training_samples[postura][ind1][0]
    #end_samp = training_samples[postura][ind1][1]
    win_size=256
    end_samp=start_samp+win_size

    power_chann1, freq = axs_chann1_15secs[posx,posy].psd(chann1[start_samp:end_samp], NFFT = win_size, Fs = 256, color='green')
    power_chann2, freq = axs_chann2_15secs[posx,posy].psd(chann2[start_samp:end_samp], NFFT = win_size, Fs = 256, color = 'red')
    
    axs_chann1_15secs[posx,posy].clear()
    axs_chann2_15secs[posx,posy].clear()

    start_freq = next(x for x, val in enumerate(freq) if val >= 4.0);
    end_freq = next(x for x, val in enumerate(freq) if val >= 60.0);

    start_index = np.where(freq >= start_freq)[0][0]
    end_index = np.where(freq >= end_freq)[0][0]

    power_chann1 = power_chann1[start_index:end_index]
    power_chann2 = power_chann2[start_index:end_index]
    freq = freq[start_index:end_index]

    axs_chann1_15secs[posx,posy].plot(freq, power_chann1, color='green')
    axs_chann2_15secs[posx,posy].plot(freq, power_chann2, color = 'red')
    
    axs_chann1_15secs[posx,posy].set_title("PSD "+str(postura) + " "+str(ind1)+" , "+str(0))
    axs_chann2_15secs[posx,posy].set_title("PSD "+str(postura) + " "+str(ind1)+" , "+str(0))
    axs_chann1_15secs[posx,posy].set(xlabel='Frecuencia (Hz)', ylabel='Power')
    axs_chann2_15secs[posx,posy].set(xlabel='Frecuencia (Hz)', ylabel='Power')


# Plot data

fig_chann1_15, axs_chann1_15secs = plt.subplots(3,2)
fig_chann1_15.suptitle('Canal 1 - Gráficas de 1 segundos')

fig_chann2_15, axs_chann2_15secs = plt.subplots(3,2)
fig_chann2_15.suptitle('Canal 2 - Gráficas de 1 segundos')

#15 secs

#VENTANA [101][1][0]
graph15secs(101,1,0,0)
psd15secs(101,1,0,1)

#VENTANA [102][0][0]
graph15secs(102,0,1,0)
psd15secs(102,0,1,1)

#VENTANA [103][2][0]
#graph15secs(103,2,2,0)
#psd15secs(103,2,2,1)


fig_chann1_15.tight_layout(pad=0.5) 
fig_chann2_15.tight_layout(pad=0.5)

fig_chann1_15.set_size_inches((14, 9.5))
fig_chann2_15.set_size_inches((14, 9.5))

plt.show()

#Graficas ventanas channel2

# Power Spectral Density (PSD) (1 second of training data)
win_size = 256

powers = []
finals = []

for chann in channels:
    for postura in posturas:
        for i in range(len(training_samples[postura])):
            for j in range(training_samples[postura][i][0],training_samples[postura][i][1],win_size):
                ini_samp = j
                end_samp = j + win_size
                x = chann[ini_samp : end_samp]
                t = time[ini_samp : end_samp]
                power, freq = plt.psd(x, NFFT = win_size, Fs = samp_rate)
                plt.clf()

                start_freq = next(x for x, val in enumerate(freq) if val >= 4.0);
                end_freq = next(x for x, val in enumerate(freq) if val >= 60.0);

                start_index = np.where(freq >= start_freq)[0][0]
                end_index = np.where(freq >= end_freq)[0][0]

                power = power[start_index:end_index]
                freq = freq[start_index:end_index]

                powers.append(power)
        #PROMEDIO
        powers = np.mean(powers,axis=0)
        
        #print("Promedio powers: ",powers)
        #print("Frecuencia: ",freq)

        #print(len(powers))

        finals.append(powers)
        powers=[]
    #finals=[]

#titulos = ["Canal 1 - 101", "Canal 1 - 102","Canal 1 - 103","Canal 2 - 101","Canal 2 - 102","Canal 2 - 103"]

titulos=[]
for j in range(len(channels)):
    for i in range(len(posturas)):
        titulos.append("Canal "+str(j+1)+" - "+str(posturas[i]))

plt.close()
fig, axs = plt.subplots(int(len(finals)//2),2)
fig.suptitle('Gráficas PSD promedio de cada postura')
i=0
for k in range(2):
    for j in range(int(len(finals)//2)):
        axs[j,k].plot(freq, finals[i])
        if(len(titulos)>i):
            axs[j,k].set_title(titulos[i])
        axs[j,k].set(xlabel='Frecuencia (Hz)', ylabel='Power')
        axs[j,k].set(xlabel='Frecuencia (Hz)', ylabel='Power')
        i+=1

for ax in axs.flat:
    ax.set(xlabel='Hz', ylabel='Power')

for ax in axs.flat:
    ax.label_outer()

fig.tight_layout(pad=0.5)
fig.set_size_inches((14, 9.5))

plt.show() 

