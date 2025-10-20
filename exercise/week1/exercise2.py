import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

 ## Constants
# Mean vectors
MU1 = np.array([-0.5, -0.5])
MU2 = np.array([0.5, 0.5])
# Covariance matrix
SIGMA = np.array([
    [1.0, 0.0],
    [0.0, 0.5]
])
# Class probabilities
P_OMEGA_1 = 0.9
P_OMEGA_2 = 0.1

## Grid see 1 to 6 in sheet01-programming.pdf
R = np.arange(-4, 4+1e-9, .1)
x, y = np.meshgrid(R, R)
# grid = np.stack((x.flatten(), y.flatten()), axis=1)
grid = list(zip(x.flatten(), y.flatten()))
# pos = np.dstack((x, y))

## MATH
# Create the Gaussian distributions
rv1 = multivariate_normal(MU1, SIGMA)
rv2 = multivariate_normal(MU2, SIGMA)

p_x_given_omega1 = multivariate_normal.pdf(grid, mean=MU1, cov=SIGMA)
p_x_given_omega2 = multivariate_normal.pdf(grid, mean=MU2, cov=SIGMA)

# Compute joint probabilities
p_joint_omega1 = p_x_given_omega1 * P_OMEGA_1
p_joint_omega2 = p_x_given_omega2 * P_OMEGA_2

# Normalize
p_normalized_omega1 = p_joint_omega1 / np.sum(p_joint_omega1)
p_normalized_omega2 = p_joint_omega2 / np.sum(p_joint_omega2)

# Reshape for plotting
p_normalized_omega1_reshaped = p_normalized_omega1.reshape(x.shape)
p_normalized_omega2_reshaped = p_normalized_omega2.reshape(x.shape)

# p_x_given_omega1_reshaped = p_x_given_omega1
# p_x_given_omega2_reshaped = p_x_given_omega2

# a) Evaluate the distributions on the grid
# p_x_given_omega1_reshaped = rv1.pdf(pos)
# p_x_given_omega2_reshaped = rv2.pdf(pos)

# Reshape the probability density functions for plotting
# p_x_given_omega1_reshaped = p_x_given_omega1.reshape(x.shape)
# p_x_given_omega2_reshaped = p_x_given_omega2.reshape(x.shape)

# b) Total probability distribution
P_x = P_OMEGA_1 * p_normalized_omega1_reshaped + P_OMEGA_2 * p_normalized_omega2_reshaped

# c) Posterior probabilities
P_omega1_given_x = (P_OMEGA_1 * p_normalized_omega1_reshaped) / P_x
P_omega2_given_x = (P_OMEGA_2 * p_normalized_omega2_reshaped) / P_x

def _ax():
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_zlim(0, 0.0020)  # Set z-axis limits from 0 to 0.0020
    return ax

def show():
    plt.tight_layout()
    plt.show()

def a():
    ax = _ax()
    # Plotting
    # ax.scatter(x, y, p_x_given_omega1_reshaped, s=1, color='blue', alpha=0.5)
    # ax.scatter(x, y, p_x_given_omega2_reshaped, s=1, color='red', alpha=0.5)

    ax.scatter(x, y, p_normalized_omega1_reshaped, s=1, color='blue', alpha=0.5)
    ax.scatter(x, y, p_normalized_omega2_reshaped, s=1, color='red', alpha=0.5)

    show()   

def b():
    ax = _ax()
    ax.scatter(x, y, P_x, s=1, alpha=0.5)
    show()

def c():
    ax = _ax()
    ax.scatter(x, y, P_omega1_given_x, s=1, color='blue', alpha=0.5)
    ax.scatter(x, y, P_omega2_given_x, s=1, color='red', alpha=0.5)

    # Bayes decision: Choose the class with higher posterior probability
    decision = P_omega1_given_x > P_omega2_given_x

    # True class labels (for error rate calculation)
    ## TODO:: This seems to be not correct, or the value is different than in the exercise sheet
    true_class_omega1 = p_normalized_omega1_reshaped > p_normalized_omega2_reshaped

    # Bayes error rate
    error_rate = np.mean(decision != true_class_omega1)
    print(f"Bayes Error Rate: {error_rate:.3f}")

    show()