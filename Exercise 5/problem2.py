import numpy as np
from scipy.special import gamma
from math import sqrt, pi

######2a######

# Define the parameters
n = 10
alpha = np.zeros(n)
beta = np.arange(n)

# Create the tridiagonal matrix A
A = np.diag(alpha) + np.diag(np.sqrt(beta[1:]), -1) + np.diag(np.sqrt(beta[1:]), 1)

# Set print options
np.set_printoptions(linewidth=np.inf)

# Print the matrix A
print("Matrix A:")
print(A)

# Solve the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# The nodes of the quadrature rule are the eigenvalues of A
nodes = eigenvalues

# The corresponding quadrature weights are the square of the first element of the eigenvectors
weights = eigenvectors[0, :]**2

print("\n-----------2a-----------\n")
# Print the nodes and weights
print("Nodes and Weights:")
for i in range(n):
    print(f"Node {i+1}: {nodes[i]}, Weight {i+1}: {weights[i]}")



######2b######
print("\n-----------2b-----------\n")
# Compute Q2k and compare with the analytical solution
print("Absolute difference |I2k - Q2k|:")
for k in range(10):
    # Analytical solution
    I2k = ((2**k)/sqrt(pi)) * gamma(k + 0.5)

    # Quadrature rule
    Q2k = np.sum(weights * nodes**(2*k))

    # Print the absolute difference
    print(f"For k = {k}: {abs(I2k - Q2k)}")

# For k = 10
k = 10
I2k = ((2**k)/sqrt(pi)) * gamma(k + 0.5)
Q2k = np.sum(weights * nodes**(2*k))
print(f"\nFor k = {k}: {abs(I2k - Q2k)}")

#explanation for k=10
print("The absolute difference |I2k - Q2k| for k = 10 is significantly larger than for k = 0, 1, ..., 9. \nThis is because the quadrature rule with n = 10 nodes and weights is only exact for polynomials of \ndegree up to 2n - 1 = 19. When k = 10, the degree of the polynomial x^(2k) = x^20 exceeds this limit, \nso the quadrature rule might not provide an accurate approximation.")