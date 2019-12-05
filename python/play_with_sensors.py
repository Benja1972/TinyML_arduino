import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import local 
from signals_processing import *

## == Load data
filename = "flex.csv"
df = pd.read_csv("../data/" + filename)
dta = 40e-3
num_samples = 119

labels = {"aX":[-4,4], "aY":[-4,4],"aZ":[-4,4], "gX":[-2000,2000], "gY":[-2000,2000],"gZ":[-2000,2000]}

smpl = Sample_arduino(df,0,num_samples,labels=labels)

idx = 3
df =  df.iloc[idx*num_samples:(idx+1)*num_samples]


## == Signal to investigate

labs = [["aX", "aY","aZ"], ["gX", "gY","gZ"]]
names = [["Accel", "Accel", "Accel"], ["Gyro","Gyro","Gyro"]]
colors = [["r","g", "b"], ["r","g", "b"]]


for j, lab in enumerate(labs):

    plt.figure()
    for i, lb in enumerate(lab):
        y = np.asarray(df[lb])
        sy = Signal_arduino(y,dta, label=lb)

        plt.plot(sy.t, sy.y,  linestyle='-', color=colors[j][i], label=lb)
        plt.title(names[j][i])
        plt.xlabel("Time")
    plt.legend()

    plt.figure()
    for i, lb in enumerate(lab):
        y = np.asarray(df[lb])
        sy = Signal_arduino(y,dta, label=lb)
        
        plt.plot(*sy.fft(), linestyle='-', color=colors[j][i], label=lb)
        plt.scatter(*get_peaks(*sy.fft()), color=colors[j][i])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.title("FFT "+names[j][i])
    plt.legend()


    plt.figure()
    for i, lb in enumerate(lab):
        y = np.asarray(df[lb])
        sy = Signal_arduino(y,dta, label=lb)
        
        plt.plot(*sy.psd(), linestyle='-', color=colors[j][i], label=lb)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2 / Hz]')
        plt.title('PSD '+ names[j][i])
    plt.legend()


    plt.figure()
    for i, lb in enumerate(lab):
        y = np.asarray(df[lb])
        sy = Signal_arduino(y,dta, label=lb)
        
        plt.plot(*sy.autocorr(), linestyle='-', color=colors[j][i], label=lb)
        plt.xlabel('time delay [s]')
        plt.ylabel('Autocorrelation amplitude')
        plt.title('Autocorr '+ names[j][i])
    plt.legend()





plt.show()



############################# TEMP #############################



#~ f, t, Sxx = signal.spectrogram(y, fs=f_s, nperseg = 50, noverlap=30)

#~ plt.figure()
#~ plt.pcolormesh(t, f, Sxx)
#~ #powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(y, Fs=0.1)
#~ plt.ylabel('Frequency [Hz]')
#~ plt.xlabel('Time [sec]')




#~ dec = sm.tsa.seasonal_decompose(y, freq=10)
#~ fig = dec.plot()
