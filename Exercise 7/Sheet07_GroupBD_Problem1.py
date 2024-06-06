import numpy as np
import matplotlib.pyplot as plt

### Problem 1 ###

def V(x):
    return ((x-4.)**2 - 2.)**2      

def dV(x):
    return 4*((x-4.)**2-2)*(x-4)


#---------a-----------

x = np.arange(1,7,0.02)
pi = np.exp(-V(x))

plt.figure(figsize=(6,5))
plt.plot(x,pi)
plt.title("Plot of π(x)")
plt.xlabel("x")
plt.grid()
plt.show()


#----------b-----------

N = 10000      #10 000 Monte-Carlo simulations
dt = 0.01
x0 = 1.0
T = 100

iteration = int(T/dt)

X = x0*np.ones(N)

for t in range(iteration):
    X = X - dV(X)*dt + np.sqrt(2*dt)*np.random.normal(size=N) 

#---------c----------

# Estimate the normalization constant C
C = 1. / (np.sum(pi) * (x[1] - x[0]))

# Scale the density function
pi_scaled = C * pi

# Plot the histogram of the simulated samples
plt.figure(figsize=(6,5))
plt.hist(X, bins=50, density=True, alpha=0.5, label='Histogram of samples')

# Plot the scaled density function
plt.plot(x, pi_scaled, label='Scaled π(x)')
plt.title("Comparison of Histogram and Scaled π(x)")
plt.xlabel("x")
plt.legend()
plt.grid()
plt.show()

print(f"The value used for C is: {C}")


#---------d----------

proportion = ( np.sum(X > 6) * 100 ) / N
print("The estimated proportion of animals which have height greater than 6 is: {:4.2f}%".format(proportion))
