import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
dt = 0.01
num_steps = 2000
time = np.arange(0, num_steps*dt, dt)

# Define the model matrices
# State transition matrix
M = np.array([[1, dt],
              [0, 1]])

# Process noise covariance
Σ = np.array([[1/4*dt**4, 1/2*dt**3],
              [1/2*dt**3, dt**2]])

# Observation model
H = np.array([[1, 0]])

# Observation noise variance
Γ = np.array([[1]])

# Initial state
x0 = np.array([0, 0])
C0 = np.array([[0, 0],
               [0, 0]])

# Simulate the true trajectory and noisy measurements
true_trajectory = 0.1 * (time**2 - time)
noisy_measurements = true_trajectory + np.random.normal(0, np.sqrt(Γ[0,0]), num_steps)

# Initialize the Kalman filter variables
x_est = x0
C_est = C0

# Storage for estimates
x_estimates = np.zeros((num_steps, 2))
x_estimates[0] = x_est

# Kalman filter implementation
for k in range(1, num_steps):
    # Prediction step
    x_pred = M @ x_est
    C_pred = M @ C_est @ M.T + Σ

    # Update step
    y_k = noisy_measurements[k]
    S = H @ C_pred @ H.T + Γ
    K = C_pred @ H.T @ np.linalg.inv(S)
    x_est = x_pred + K @ (y_k - H @ x_pred)
    C_est = (np.eye(2) - K @ H) @ C_pred

    # Store estimates
    x_estimates[k] = x_est

# Extract position and velocity estimates
positions = x_estimates[:, 0]
velocities = x_estimates[:, 1]

# Plotting the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, positions, label='Filtered Position')
plt.plot(time, true_trajectory, label='True Position', linestyle='dashed')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, velocities, label='Filtered Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.legend()

plt.tight_layout()
plt.show()

