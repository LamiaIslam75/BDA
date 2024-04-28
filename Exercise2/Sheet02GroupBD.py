# Problem 1 ------- Lamia Islam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# parameters
dt = 0.01
n_cycle = 5   #save data on each 5 time steps
n_total = 4001  #total number of data in order to compute until t = 200
z0 = np.array([-0.587, -0.563, 16.870])

#generating noise with tent map
a = dt**(-1/2)
def tmap(g):
    if (g>=-a/2) & (g<0):
        g = 1.99999*g + a/2
    elif (g>=0) & (g<=a/2):
        g = -1.99999*g + a/2
    else:
        #make sure it does not blow up
        print('out of range')

    return g

#Simulate Lorenz63 system

z = np.copy(z0)     #running value
g1, g2, g3 = a*(2**(-1/2)-1/2),a*(3**(-1/2)-1/2),a*(5**(-1/2)-1/2)
z_out = np.zeros((n_total,3))
for i in range(n_total):
    z_out[i] = z
    for j in range(n_cycle):
        g1, g2, g3 = tmap(g1), tmap(g2), tmap(g3)
        z += np.array([10*(z[1]-z[0]) + g1,z[0]*(28-z[2])-z[1] +g2, z[0]*z[1]-8*z[2]/3 +g3])*dt

e_t = np.zeros(n_total)

e = a*(2**(-1/2)-1/2)

for i in range(n_total):
    for j in range(20):
        e_t[i] += e/20
        for k in range(10):
            e = tmap(e) 

y_out = z_out[:,0]+e_t
t = (np.cumsum(np.ones(n_total))-1)*dt*n_cycle

# part a

plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(t[:200],y_out[:200],'x')
plt.xlabel('Time')
plt.ylabel('x-coordinate')
plt.title('Observed Values')

plt.subplot(2, 1, 2)
plt.plot(e_t[:200],'x')
plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Measurement Error')
plt.tight_layout()
plt.show()

# part b

np.savetxt('reference_trajectory.txt', z_out)

print('Problem 1:')
print(f'Mean of x-component: {np.mean(z_out[:,0]):.6f}')
print(f'Std. dev. of x-component: {np.std(z_out[:,0]):.6f}')
print(f'Mean of y-component: {np.mean(z_out[:,1]):.6f}')
print(f'Std. dev. of y-component: {np.std(z_out[:,1]):.6f}')
print(f'Mean of z-component: {np.mean(z_out[:,2]):.6f}')
print(f'Std. dev. of z-component: {np.std(z_out[:,2]):.6f}')
print('---------------------------------------------------------------')

# ---------------------------------------------------------------

# Problem 2 ------- Md Borhan Uddin

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
print('Problem 2:')
print(f'Mean of observation values: {mean_obs:.6f}')
print(f'Std. dev. of observation values: {std_obs:.6f}')
print('---------------------------------------------------------------')
# ---------------------------------------------------------------

##Problem 3 - Sanjay Rajpurohit

observations = xobs
time = t

# Linear extrapolation
def linear_extrapolation(t, obs, delta_t):
    extrapolated = np.zeros_like(obs)
    extrapolated[:-1] = obs[1:]  # Initialize with shifted observations
    
    for i in range(len(obs) - 1):
        slope = (obs[i + 1] - obs[i]) / (t[i + 1] - t[i])
        extrapolated[i + 1] = obs[i + 1] + slope * delta_t
    
    return extrapolated


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
plt.plot(time[:-1] + delta_t1, forecast1[:-1], 'r--', label=f'Forecast ∆t={delta_t1}')
plt.plot(time[:-1] + delta_t3, forecast3[:-1], 'g--', label=f'Forecast ∆t={delta_t3}')
plt.xlim([100, 105])
plt.xlabel('Time')
plt.ylabel('Observations and Forecasts')
plt.legend()
plt.grid(True)
plt.show()

# Compute time-averaged RMSE
rmse1 = np.sqrt(mean_squared_error(observations, forecast1))
rmse3 = np.sqrt(mean_squared_error(observations, forecast3))

print('Problem 3:')
print(f'Time averaged RMSE for ∆t={delta_t1:.2f}: {rmse1:.6f}')
print(f'Time averaged RMSE for ∆t={delta_t3:.2f}: {rmse3:.6f}')