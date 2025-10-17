# solutions.py file for the hw1
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Exercise 1 ----------------
def s1a():
    # Create grid from -4 to 4 with step 0.1
    x = np.arange(-4, 4.1, 0.1) # example에서 arrange 가 -4 ~0.1 이라
    y = np.arange(-4, 4.1, 0.1)
    X, Y = np.meshgrid(x, y)
    
    # Compute 2D Gaussian distribution
    Z_vals = np.exp(-0.5*(X**2 + Y**2))        # e^{-0.5(x^2+y^2)}
    Z_normalized = Z_vals / Z_vals.sum()       # (1/Z) * e^{-0.5(x^2+y^2)}

    # Plot the distribution
    plt.figure(figsize=(6,5))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title('Discretized Gaussian P(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return X, Y, Z
