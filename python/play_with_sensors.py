import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filename = "flex.csv"

df = pd.read_csv("../data/" + filename)

index = range(1, len(df['aX']) + 1)

plt.rcParams["figure.figsize"] = (20,10)

plt.figure()
plt.plot(index, df['aX'], 'g.', label='x', linestyle='solid', marker=',')
plt.plot(index, df['aY'], 'b.', label='y', linestyle='solid', marker=',')
plt.plot(index, df['aZ'], 'r.', label='z', linestyle='solid', marker=',')
plt.title("Acceleration")
plt.xlabel("Sample #")
plt.ylabel("Acceleration (G)")
plt.legend()

plt.figure()
plt.plot(index, df['gX'], 'g.', label='x', linestyle='solid', marker=',')
plt.plot(index, df['gY'], 'b.', label='y', linestyle='solid', marker=',')
plt.plot(index, df['gZ'], 'r.', label='z', linestyle='solid', marker=',')
plt.title("Gyroscope")
plt.xlabel("Sample #")
plt.ylabel("Gyroscope (deg/sec)")
plt.legend()




import statsmodels.api as sm
y = np.asarray(df['aX'])

dec = sm.tsa.seasonal_decompose(y, freq=10)
fig = dec.plot()



from scipy import signal

f, t, Sxx = signal.spectrogram(y, fs=1, nperseg = 50, noverlap=30)

plt.figure()
plt.pcolormesh(t, f, Sxx)
#powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(y, Fs=0.1)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
