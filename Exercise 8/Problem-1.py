import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats

def gibbs_sampler(p, N=10000):
    x1, x2 = 0.0, 0.0
    samples = np.zeros((N, 2))

    for i in range(N):
        x1 = stats.norm.rvs(loc=p*x2, scale=np.sqrt(1-p**2))
        x2 = stats.norm.rvs(loc=p*x1, scale=np.sqrt(1-p**2))
        samples[i, :] = [x1, x2]

    return samples

def estimate_and_print(samples):
    # Estimate the expected values
    expected_x1 = np.mean(samples[:, 0])
    expected_x2 = np.mean(samples[:, 1])

    # Estimate the standard deviations
    std_dev_x1 = np.std(samples[:, 0])
    std_dev_x2 = np.std(samples[:, 1])

    print(f"Expected Values: x1={expected_x1}, x2={expected_x2}")
    print(f"Standard Deviations: x1={std_dev_x1}, x2={std_dev_x2}")

def plot_samples(samples, p):
    # Plotting
    plt.figure(figsize=(10, 10))

    # Scatter plot of samples
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, label='Samples')

    # Contour plot of the probability density function
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = stats.multivariate_normal.pdf(np.dstack((X, Y)), mean=[0, 0], cov=[[1, p], [p, 1]])
    plt.contour(X, Y, Z, colors='r')
    
    # Create a legend for the contour plot
    red_patch = mpatches.Patch(color='red', label='PDF')
    plt.legend(handles=[red_patch])

    # Set title and labels
    plt.title(f"Gibbs Sampler for p={p} against a Contour Plot of the Probability Density π")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

# Part (a)
print("--------------a---------------")
p = 0.5
samples = gibbs_sampler(p)
estimate_and_print(samples)
plot_samples(samples, p)

# Part (b)
print("--------------b---------------")
p_values = [0.5, 0.9, 0.99, 0.999]
for p in p_values:
    samples = gibbs_sampler(p)
    print(f"For p={p}:")
    estimate_and_print(samples)
    plot_samples(samples, p)
