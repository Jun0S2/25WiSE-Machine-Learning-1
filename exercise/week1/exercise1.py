import numpy as np
import matplotlib.pyplot as plt

R = np.arange(-4, 4+1e-9, .1)
X, Y = np.meshgrid(R, R)
# grid = np.stack((x.flatten(), y.flatten()), axis=1)
grid = list(zip(X.flatten(), Y.flatten()))

## a)
# Compute unnormalized distribution
unnormalized = np.exp(-0.5 * (X**2 + Y**2))
# Compute partition function Z
Z = np.sum(unnormalized)
# Normalize
P = unnormalized / Z

## b)
# Compute condition
mask = (X**2 + Y**2) >= 1
# Compute unnormalized conditional distribution
unnormalized_conditional = np.where(mask, P, 0)
# Compute new partition function Z'
Z = np.sum(unnormalized_conditional)
# Normalize
Q = unnormalized_conditional / Z

## c)
# Marginalize over y
Q_x = np.sum(Q, axis=0)

def fig():
    return plt.figure(figsize=(10, 6))

def show():
    plt.tight_layout()
    plt.show()

def a():
    ax = fig().add_subplot(111, projection='3d')
    ax.scatter(X, Y, P, s=1, alpha=0.5)
    show()

def b():
    ax = fig().add_subplot(111, projection='3d')
    ax.scatter(X, Y, Q, s=1, alpha=0.5)
    show()

def c():
    ax = fig().add_subplot(111)
    ax.plot(R, Q_x, 'o', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('Q(x)')
    show()


# a()