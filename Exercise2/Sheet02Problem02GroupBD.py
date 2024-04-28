# Problem 2 ------- Md Borhan Uddin

import numpy as np
import matplotlib.pyplot as plt

# Load reference trajectory
z_out = np.loadtxt('reference_trajectory.txt')

# Parameters
dt = 0.01
n_total = 4001
a = 4

# Generate noise using tent map
def tmap(g):
    if (g >= -a/2) and (g < 0):
        g = 1.99999 * g + a/2
    elif (g >= 0) and (g <= a/2):
        g = -1.99999 * g + a/2
    else:
        print('Out of range')
    return g

# Generate observation values
xobs = np.zeros(n_total)
e = a * (2**(-1/2) - 1/2)
for k in range(n_total):
    for i in range(20):
        xobs[k] += e / 20
        for j in range(10):
            e = tmap(e)
    xobs[k] += z_out[k, 0]

# Plot observed x-components and measurement errors
t = np.arange(0, n_total * dt, dt)
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(t[:200], xobs[:200], 'x')
plt.xlabel('Time')
plt.ylabel('x-coordinate')
plt.title('Observed Values')

plt.subplot(2, 1, 2)
plt.plot(t[:200], xobs[:200] - z_out[:200, 0], 'x')
plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Measurement Error')
plt.tight_layout()
plt.show()

# Store observation values in a file
np.savetxt('observation_values.txt', xobs)

# Calculate mean and standard deviation of observation values
mean_obs = np.mean(xobs)
std_obs = np.std(xobs)
print(f'Mean of observation values: {mean_obs:.6f}')
print(f'Std. dev. of observation values: {std_obs:.6f}')



##Problem 3 - Sanjay Rajpurohit


observations = xobs
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
