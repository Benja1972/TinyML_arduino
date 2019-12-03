import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# signal processing
import statsmodels.api as sm
from scipy.fftpack import fft
from scipy.signal import welch
from scipy import signal


class Signal_arduino:
    def __init__(self, y, T, label=""):
        self.y = y
        self.T = T
        self.N = len(y)
        self.dt = T/self.N
        self.fs = 1/self.dt
        self.t = self.dt*np.arange(0, self.N)
        self.label=label

    def fft(self):
        # fft trafsform
        f = np.linspace(0.0, 1.0/(2.0*self.dt), self.N//2)
        fft_vals_ = fft(self.y)
        fft_vals = 2.0/self.N * np.abs(fft_vals_[0:self.N//2])
        return f, fft_vals

    def psd(self):
        f, psd_vals = welch(self.y, fs=self.fs)
        return f, psd_vals

    def autocorr(self):
        res = np.correlate(self.y, self.y, mode='full')
        autocorr_vals = res[len(res)//2:]
        return self.t, autocorr_vals

# wavelet -- TODO!!!


## == Load data
filename = "flex.csv"
df = pd.read_csv("../data/" + filename)

## == Signal to investigate

labs = ["aX", "aY","aZ"]
colors = ["r","g", "b"]

plt.figure()
for i, lb in enumerate(labs):
    lb = labs[i]
    y = np.asarray(df[lb])
    sy = Signal_arduino(y,T=20, label=lb)

    plt.plot(sy.t, sy.y,  linestyle='-', color=colors[i], label=lb)
    plt.title("Acceleration ")
    plt.xlabel("Time")
plt.legend()

plt.figure()
for i, lb in enumerate(labs):
    lb = labs[i]
    y = np.asarray(df[lb])
    sy = Signal_arduino(y,T=20, label=lb)
    
    plt.plot(*sy.fft(), linestyle='-', color=colors[i], label=lb)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title("FFT Accel")
plt.legend()


plt.figure()
for i, lb in enumerate(labs):
    lb = labs[i]
    y = np.asarray(df[lb])
    sy = Signal_arduino(y,T=20, label=lb)
    plt.plot(*sy.psd(), linestyle='-', color=colors[i], label=lb)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2 / Hz]')
    plt.title('PSD Accel')
plt.legend()


plt.figure()
for i, lb in enumerate(labs):
    lb = labs[i]
    y = np.asarray(df[lb])
    sy = Signal_arduino(y,T=20, label=lb)
    plt.plot(*sy.autocorr(), linestyle='-', color=colors[i], label=lb)
    plt.xlabel('time delay [s]')
    plt.ylabel('Autocorrelation amplitude')
    plt.title('Autocorr Accel')
plt.legend()





plt.show()



############################# TEMP #############################
#plt.rcParams["figure.figsize"] = (20,10)

#~ plt.figure()
#~ plt.plot(index, df['aX'], 'g.', label='x', linestyle='solid', marker=',')
#~ plt.plot(index, df['aY'], 'b.', label='y', linestyle='solid', marker=',')
#~ plt.plot(index, df['aZ'], 'r.', label='z', linestyle='solid', marker=',')
#~ plt.title("Acceleration")
#~ plt.xlabel("Sample #")
#~ plt.ylabel("Acceleration (G)")
#~ plt.legend()

#~ plt.figure()
#~ plt.plot(index, df['gX'], 'g.', label='x', linestyle='solid', marker=',')
#~ plt.plot(index, df['gY'], 'b.', label='y', linestyle='solid', marker=',')
#~ plt.plot(index, df['gZ'], 'r.', label='z', linestyle='solid', marker=',')
#~ plt.title("Gyroscope")
#~ plt.xlabel("Sample #")
#~ plt.ylabel("Gyroscope (deg/sec)")
#~ plt.legend()



#~ f, t, Sxx = signal.spectrogram(y, fs=f_s, nperseg = 50, noverlap=30)

#~ plt.figure()
#~ plt.pcolormesh(t, f, Sxx)
#~ #powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(y, Fs=0.1)
#~ plt.ylabel('Frequency [Hz]')
#~ plt.xlabel('Time [sec]')




#~ dec = sm.tsa.seasonal_decompose(y, freq=10)
#~ fig = dec.plot()
