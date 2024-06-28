###
# Sheet 10 Group BD
# Problem 1 - Lamia Islam
###

import numpy as np
import matplotlib.pyplot as plt

# Parameters
delta_t = 0.01
M = np.array([[1, delta_t], [0, 1]])
Sigma = np.array([[1/4 * delta_t**4 + 1e-10, 1/2 * delta_t**3],
                  [1/2 * delta_t**3, delta_t**2 + 1e-10]])
H = np.array([[1, 0]])
Gamma = np.array([[1]])
N = 100  # Ensemble size
timesteps = 2000

# True trajectory function
def true_trajectory(t):
    return 0.1 * (t**2 - t)

# Initialize ensemble
x_ensemble = np.zeros((N, 2))

# Generate true trajectory and noisy measurements
t = np.arange(0, timesteps * delta_t, delta_t)
true_positions = true_trajectory(t)
measurements = true_positions + np.random.normal(0, 1, timesteps)

# Store the estimated means
estimated_positions = np.zeros(timesteps)
estimated_velocities = np.zeros(timesteps)

# Ensemble Kalman Filter
for k in range(timesteps):
    # Prediction step
    for i in range(N):
        x_ensemble[i] = M @ x_ensemble[i] + np.random.multivariate_normal([0, 0], Sigma)
    
    # Compute ensemble mean and covariance
    x_mean = np.mean(x_ensemble, axis=0)
    P = np.cov(x_ensemble.T)
    
    # Compute Kalman gain
    S = H @ P @ H.T + Gamma
    K = P @ H.T @ np.linalg.inv(S)
    
    # Update step
    y_k = measurements[k]
    for i in range(N):
        x_ensemble[i] = x_ensemble[i] + K @ (y_k - H @ x_ensemble[i])
    
    # Store results
    estimated_positions[k] = np.mean(x_ensemble[:, 0])
    estimated_velocities[k] = np.mean(x_ensemble[:, 1])

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, estimated_positions, label="Estimated Position")
plt.plot(t, true_positions, label="True Position", linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, estimated_velocities, label="Estimated Velocity")
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()

plt.tight_layout()
plt.show()
