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
dt = 5
#input_data = df[['T_bat', 'current', 'omega_scaled', 'Q_heat_scaled']].to_numpy()
input_data = df[['T_bat']].to_numpy().flatten()
current_data =df[['current']].to_numpy().flatten()
#input_data = df[['h1', 'h2', 'u']].to_numpy()
print(input_data)
#residuals = df[['residual_1', 'residual_2']].to_numpy()
residuals = df[['residual']].to_numpy().flatten()
T_bat_dot_model = df[['T_bat_dot_model']].to_numpy().flatten()

print(input_data.shape)
print(residuals)
polyorder = 3
window_length = 20
residuals_filtered = savgol_filter(residuals, window_length=window_length, polyorder=polyorder)
print("len res", len(residuals))
print("len fil", len(residuals_filtered))
plt.figure(1)
plt.plot(residuals)
plt.plot(residuals_filtered)
plt.legend(('residuals', 'savgol filtered'), loc='best')

plt.figure(2)
plt.plot(input_data)
temperatures_filtered = savgol_filter(input_data, window_length=window_length, polyorder=polyorder)
filter_order = 5
fs = 2*np.pi*1/5
w_critical = 0.2 # normalized frequency to nyquist frequency (half of sample rate)
# b, a
numerator_coeffs, denominator_coeffs = signal.butter(filter_order, w_critical, fs=fs)
temperatures_bw_filtered = signal.filtfilt(numerator_coeffs, denominator_coeffs, input_data)
 
plt.plot(temperatures_filtered)
plt.plot(temperatures_bw_filtered)
plt.legend(('temperatures', 'savgol filtered', 'bw filter'), loc='best')

plt.figure(3)
derivative_of_filtered_temperatures = np.diff(temperatures_filtered)/dt
derivative_of_bw_filtered_temperatures = np.diff(temperatures_bw_filtered)/dt
derivative_of_temperatures = np.diff(input_data)/dt
T_bat_dot_model_filtered = savgol_filter(T_bat_dot_model, window_length=window_length, polyorder=polyorder)
T_bat_dot_model_bw_filtered = signal.filtfilt(numerator_coeffs, denominator_coeffs, T_bat_dot_model)
print(temperatures_filtered)
plt.plot(derivative_of_temperatures)
plt.plot(derivative_of_filtered_temperatures)
plt.plot(derivative_of_bw_filtered_temperatures)

plt.plot(T_bat_dot_model)
plt.plot(T_bat_dot_model_bw_filtered)
print("len temperatures", len(temperatures_filtered))
print("len derivative", len(derivative_of_filtered_temperatures))
plt.legend(('temperatures deriv', 'deriv savgol filtered', 'bw filtered', "T_bat_dot_model", "T_bat_dot_filtered"), loc='best')

#plt.figure(4)
#plt.plot(T_bat_dot_model)
#plt.plot(T_bat_dot_model_filtered)

plt.figure(4)
plt.plot(derivative_of_temperatures[:-1]-T_bat_dot_model[2:])
plt.plot(derivative_of_filtered_temperatures[:-1]-T_bat_dot_model_filtered[2:])
plt.plot(derivative_of_bw_filtered_temperatures[:-1]-T_bat_dot_model_bw_filtered[2:])

plt.plot(residuals_filtered)
plt.legend(('residuals no filtering', 'residuals filter before', "residuals bw filter before", "residuals filtered after"), loc='best')

plt.show()
"""
plt.figure(2)
filter_order = 5
fs = 2*np.pi*1/5
w_critical = 0.05 # normalized frequency to nyquist frequency (half of sample rate)
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
plt.figure(4)
#filter_order = 5
#fs = 2*np.pi*1/5
#w_critical = 0.05 # normalized frequency to nyquist frequency (half of sample rate)
# b, a
#numerator_coeffs, denominator_coeffs = signal.butter(filter_order, w_critical, fs=fs)
print(numerator_coeffs)
current_filtered_butterworth = signal.filtfilt(numerator_coeffs, denominator_coeffs, current_data)
plt.plot(current_data)
plt.plot(current_filtered_butterworth)
plt.legend(('current', 'butterworth filtered'), loc='best')
print(len(residuals))
print(len(residual_filtered_butterworth))
plt.show()
"""