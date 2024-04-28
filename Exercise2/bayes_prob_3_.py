

##Problem 3 - Sanjay Rajpurohit

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

y_obs = np.genfromtxt('observation_values.txt', delimiter=',')
N = len(y_obs)

# Linear extrapolation
def linear_extrapolation(t, obs, delta_t):
    slope = (obs[-1] - obs[-2]) / (t[-1] - t[-2])
    return obs[-1] + slope * delta_t

# Forecast intervals
delta_t1 = 0.05
delta_t3 = 0.15

# Forecasts
forecast1 = linear_extrapolation(time, observations, delta_t1)
forecast3 = linear_extrapolation(time, observations, delta_t3)

# Time interval for plot
t_interval = np.linspace(100, 105, 100)

# Plotting it
plt.figure(figsize=(10, 6))
plt.plot(time, observations, label='Observations')
plt.plot(time[-1] + delta_t1, forecast1, 'ro', label=f'Forecast ∆t={delta_t1}')
plt.plot(time[-1] + delta_t3, forecast3, 'go', label=f'Forecast ∆t={delta_t3}')
plt.xlim([100, 105])
plt.xlabel('Time')
plt.ylabel('Observations and Forecasts')
plt.legend()
plt.grid(True)
plt.show()

# Compute time averaged RMSE
rmse1 = np.sqrt(mean_squared_error(observations[-int(1/delta_t1):], [forecast1]*int(1/delta_t1)))
rmse3 = np.sqrt(mean_squared_error(observations[-int(1/delta_t3):], [forecast3]*int(1/delta_t3)))

print(f'Time averaged RMSE for ∆t={delta_t1}: {rmse1}')
print(f'Time averaged RMSE for ∆t={delta_t3}: {rmse3}')

