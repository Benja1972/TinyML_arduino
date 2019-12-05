import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# signal processing
#import statsmodels.api as sm
from scipy.fftpack import fft
from scipy.signal import welch
from scipy import signal
from detect_peaks import detect_peaks

class Signal_arduino:
    def __init__(self, y, dt=1, label=""):
        self.y = y
        self.dt = dt
        self.N = len(y)
        self.T = self.N*self.dt
        self.fs = 1/self.dt
        self.t = self.dt*np.arange(0, self.N)
        self.label=label

    def fft(self):
        # fft transform
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




class Sample_arduino:
    def __init__(self, df, start, end, labels=[], dt=1):
        self.data={}
        for lb in labels:
            self.data[lb]=Signal_arduino(df.iloc[start:end][lb], dt=dt, label=lb)
        

# wavelet -- TODO!!!


def get_peaks(x,y,mph=None, n=5):
    if not mph:
        mph = 0.2*np.nanmax(y)
    ind = detect_peaks(y, mph=mph)
    xp, yp = list(x[ind]), list(y[ind])
    if len(xp) >= n:
        return xp[:n], yp[:n]
    else:
        miss = n-len(xp)
        return xp + [0]*miss, yp + [0]*miss


def mph_calc(y,perc=5, dnt=10):
    y_min = np.nanpercentile(np.abs(y), perc)
    y_max = np.nanpercentile(np.abs(y), 100-perc)
    return y_min + (y_max - y_min)/dnt





## == Load data
filename = "flex.csv"
df = pd.read_csv("../data/" + filename)
dta = 40e-3
num_samples = 119

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
