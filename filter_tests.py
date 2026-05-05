import os
import pandas as pd
from scipy.signal import savgol_filter
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
# Load dataset of residuals
csv_path = os.path.join(os.path.dirname(__file__), 'residuals.csv')
df = pd.read_csv(csv_path)
print(df.head())
#input_data = df[['T_bat', 'current', 'omega_scaled', 'Q_heat_scaled']].to_numpy()
input_data = df[['T_bat']].to_numpy()
#input_data = df[['h1', 'h2', 'u']].to_numpy()
print(input_data)
#residuals = df[['residual_1', 'residual_2']].to_numpy()
residuals = df[['residual']].to_numpy().flatten()
print(input_data.shape)
print(residuals)
polyorder = 3
window_length = 10
residuals_filtered = savgol_filter(residuals, window_length=window_length, polyorder=polyorder)
print("len res", len(residuals))
print("len fil", len(residuals_filtered))
plt.figure(1)
plt.plot(residuals)
plt.plot(residuals_filtered)
plt.legend(('residuals', 'savgol filtered'), loc='best')


plt.figure(2)
filter_order = 3
fs = 2*np.pi*1/5
w_critical = 0.1 # normalized frequency to nyquist frequency (half of sample rate)
# b, a
numerator_coeffs, denominator_coeffs = signal.butter(filter_order, w_critical, fs=fs)
print(numerator_coeffs)
residual_filtered_butterworth = signal.filtfilt(numerator_coeffs, denominator_coeffs, residuals)
plt.plot(residuals)
plt.plot(residual_filtered_butterworth)
plt.legend(('residuals', 'butterworth filtered'), loc='best')

plt.figure(3)
w, h = signal.freqz(numerator_coeffs, denominator_coeffs, fs=fs)
plt.semilogx(w, 20*np.log10(abs(h)))
plt.title("Butterworth filter frequency response")
plt.xlabel("frequency in rad/sample")
plt.ylabel("amplitude")
plt.grid(which='both', axis='both')
plt.axvline(w_critical, color='green')
plt.show()