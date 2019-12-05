import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from scipy.fftpack import fft
from scipy.signal import welch
from detect_peaks import detect_peaks

class Signal_arduino:
    def __init__(self, y, dt=1, ranges = [],label=""):
        self.y = y
        if ranges:
            self.y_norm = (self.y-ranges[0])/(ranges[1] - ranges[0])
        else:
            self.y_norm = self.y
        self.dt = dt
        self.ranges = ranges
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
    def __init__(self, df, start, end, labels={}, dt=1):
        self.data={}
        for lb, rng in labels.items():
            self.data[lb]=Signal_arduino(df.iloc[start:end][lb].values, 
                                        dt=dt, 
                                        label=lb,
                                        ranges=rng)
    
    def get_data_matrix(self):
        mtx = []
        for lb, sig in self.data.items():
            mtx.append(list(sig.y_norm))
        return np.transpose(np.array(mtx))
    
    
    def get_data_vector(self):
        return np.reshape(self.get_data_matrix(), (1,-1))
        
        

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


