# Problem 1 ------- Lamia Islam
import numpy as np
import matplotlib.pyplot as plt

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

print(f'Mean of x-component: {np.mean(z_out[:,0]):.6f}')
print(f'Std. dev. of x-component: {np.std(z_out[:,0]):.6f}')
print(f'Mean of y-component: {np.mean(z_out[:,1]):.6f}')
print(f'Std. dev. of y-component: {np.std(z_out[:,1]):.6f}')
print(f'Mean of z-component: {np.mean(z_out[:,2]):.6f}')
print(f'Std. dev. of z-component: {np.std(z_out[:,2]):.6f}')

# ---------------------------------------------------------------