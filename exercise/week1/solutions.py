import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===================== Exercise 1: Gaussian distributions =====================

def s1a():
    """
    Compute the distribution P(x,y), and plot it.
    P(x,y) = (1/Z) * exp(-0.5*(x^2 + y^2))
    where Z = sum over all grid points of exp(-0.5*(x^2 + y^2))
    """
    print("=== Exercise 1a: 2D Gaussian Distribution P(x,y) ===")
    
    # 1. Create grid from -4 to 4 with step 0.1
    R = np.arange(-4, 4 + 1e-9, 0.1)  # +1e-9 to include 4.0
    X, Y = np.meshgrid(R, R)
    print(f"Grid size: {X.shape}")
    
    # 2. Compute unnormalized Gaussian: exp(-0.5*(x^2 + y^2))
    unnormalized_Z = np.exp(-0.5 * (X**2 + Y**2))
    
    # 3. Normalize so that sum over all grid points = 1
    Z = unnormalized_Z / np.sum(unnormalized_Z)
    print(f"Sum of normalized distribution: {np.sum(Z):.6f} (should be 1.0)")
    
    # 4. Plot the distribution
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    ax.scatter(X, Y, Z, s=1, alpha=0.6, c=Z, cmap='viridis')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('P(x,y)')
    ax.set_title('2D Gaussian Distribution P(x,y)')
    
    plt.show()
    
    return X, Y, Z

def s1b():
    """
    Compute the conditional distribution P(x,y|y=1), and plot it.
    This means we fix y=1 and look at how probability varies with x.
    """
    print("\n=== Exercise 1b: Conditional Distribution P(x,y|y=1) ===")
    
    # 1. Create grid (same as before)
    R = np.arange(-4, 4 + 1e-9, 0.1)
    X, Y = np.meshgrid(R, R)
    
    # 2. Find indices where y is approximately 1
    # We need to account for floating point precision
    y_target = 1.0
    tolerance = 0.05  # Allow small tolerance
    
    # Create mask for y values close to 1
    mask = np.abs(Y - y_target) < tolerance
    
    # 3. Compute conditional distribution
    # P(x,y|y=1) = P(x,y) / P(y=1), but only defined where y≈1
    
    # First compute P(x,y) as before
    unnormalized_Z = np.exp(-0.5 * (X**2 + Y**2))
    P_xy = unnormalized_Z / np.sum(unnormalized_Z)
    
    # Extract the slice where y≈1
    X_cond = X[mask]
    Y_cond = Y[mask] 
    P_xy_cond = P_xy[mask]
    
    # Normalize this slice to sum to 1
    P_cond_normalized = P_xy_cond / np.sum(P_xy_cond)
    
    print(f"Number of points in conditional slice: {len(X_cond)}")
    print(f"Sum of conditional distribution: {np.sum(P_cond_normalized):.6f}")
    
    # 4. Plot the conditional distribution
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(X_cond, Y_cond, P_cond_normalized, s=10, alpha=0.8, 
               c=P_cond_normalized, cmap='viridis')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('P(x,y|y=1)')
    ax.set_title('Conditional Distribution P(x,y|y=1)')
    
    plt.show()
    
    return X, Y, P_xy

def s1c():
    """
    Marginalize the conditioned distribution P(x,y|y=1) over y, 
    and plot the resulting distribution P(x|y=1).
    
    Since we conditioned on y=1, marginalizing over y gives us P(x|y=1).
    """
    print("\n=== Exercise 1c: Marginal Distribution P(x|y=1) ===")
    
    # 1. Create grid
    R = np.arange(-4, 4 + 1e-9, 0.1)
    X, Y = np.meshgrid(R, R)
    
    # 2. Get y=1 slice (same as in 1b)
    y_target = 1.0
    tolerance = 0.05
    
    # Find unique x values and their probabilities
    unique_x = []
    probs = []
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if abs(Y[i,j] - y_target) < tolerance:
                x_val = X[i,j]
                # Compute probability at this point
                prob = np.exp(-0.5 * (x_val**2 + y_target**2))
                
                # Check if we already have this x value
                found = False
                for k, existing_x in enumerate(unique_x):
                    if abs(existing_x - x_val) < 0.01:  # Small tolerance for x
                        probs[k] += prob
                        found = True
                        break
                
                if not found:
                    unique_x.append(x_val)
                    probs.append(prob)
    
    # Convert to numpy arrays and normalize
    unique_x = np.array(unique_x)
    probs = np.array(probs)
    probs = probs / np.sum(probs)  # Normalize to sum=1
    
    print(f"Number of unique x values: {len(unique_x)}")
    print(f"Sum of marginal distribution: {np.sum(probs):.6f}")
    
    # 3. Plot the marginal distribution P(x|y=1)
    plt.figure(figsize=(10, 6))
    plt.plot(unique_x, probs, 'b-', linewidth=2, label='P(x|y=1)')
    plt.fill_between(unique_x, probs, alpha=0.3)
    
    plt.xlabel('X')
    plt.ylabel('Probability')
    plt.title('Marginal Distribution P(x|y=1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()
    
    return unique_x, probs

# ===================== Exercise 2: Bayesian Classification =====================

def s2a():
    """
    Discretize the two data-generating distributions and plot them with different colors.
    """
    print("=== Exercise 2a: Data-Generating Distributions ===")
    
    # 1. Create grid
    R = np.arange(-4, 4 + 1e-9, 0.1)
    X, Y = np.meshgrid(R, R)
    
    # 2. Define parameters
    mu1 = np.array([1, 1])    # Mean for class 1
    mu2 = np.array([-1, -1])  # Mean for class 2  
    Sigma = np.array([[2, 1], [1, 2]])  # Covariance matrix
    
    # 3. Compute multivariate Gaussian distributions
    # Formula: P(x|ω) = (1/(2π|Σ|^0.5)) * exp(-0.5 * (x-μ)ᵀ Σ⁻¹ (x-μ))
    
    # Compute inverse and determinant of covariance matrix
    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_det = np.linalg.det(Sigma)
    
    # Precompute constant factor
    constant = 1 / (2 * np.pi * np.sqrt(Sigma_det))
    
    # Compute P(x|ω1) and P(x|ω2) for all grid points
    P_x_given_omega1 = np.zeros_like(X)
    P_x_given_omega2 = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_vec = np.array([X[i,j], Y[i,j]])
            
            # For class 1
            diff1 = x_vec - mu1
            exponent1 = -0.5 * diff1.T @ Sigma_inv @ diff1
            P_x_given_omega1[i,j] = constant * np.exp(exponent1)
            
            # For class 2  
            diff2 = x_vec - mu2
            exponent2 = -0.5 * diff2.T @ Sigma_inv @ diff2
            P_x_given_omega2[i,j] = constant * np.exp(exponent2)
    
    # 4. Plot both distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot P(x|ω1)
    contour1 = ax1.contourf(X, Y, P_x_given_omega1, levels=50, cmap='Reds')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('P(x|ω₁) - Class 1 Distribution')
    ax1.plot(1, 1, 'ro', markersize=10, label='Mean μ₁')
    ax1.legend()
    plt.colorbar(contour1, ax=ax1)
    
    # Plot P(x|ω2)  
    contour2 = ax2.contourf(X, Y, P_x_given_omega2, levels=50, cmap='Blues')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('P(x|ω₂) - Class 2 Distribution')
    ax2.plot(-1, -1, 'bo', markersize=10, label='Mean μ₂')
    ax2.legend()
    plt.colorbar(contour2, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    return X, Y, P_x_given_omega1, P_x_given_omega2

def s2b():
    """
    Compute the total probability distribution P(x) and plot it.
    P(x) = P(ω1)P(x|ω1) + P(ω2)P(x|ω2)
    """
    print("\n=== Exercise 2b: Total Probability Distribution P(x) ===")
    
    # Get distributions from previous function
    X, Y, P_x_given_omega1, P_x_given_omega2 = s2a()
    
    # Define class priors
    P_omega1 = 0.5
    P_omega2 = 0.5
    
    # Compute total probability: P(x) = Σ P(ωᵢ)P(x|ωᵢ)
    P_x = P_omega1 * P_x_given_omega1 + P_omega2 * P_x_given_omega2
    
    # Plot total distribution
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, P_x, levels=50, cmap='viridis')
    plt.colorbar(contour)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Total Probability Distribution P(x)')
    
    # Mark the means
    plt.plot(1, 1, 'ro', markersize=10, label='μ₁ (Class 1)')
    plt.plot(-1, -1, 'bo', markersize=10, label='μ₂ (Class 2)')
    plt.legend()
    
    plt.show()
    
    return X, Y, P_x

def s2c():
    """
    Compute class posterior probabilities and Bayes error rate.
    """
    print("\n=== Exercise 2c: Posterior Probabilities and Bayes Error ===")
    
    # Get data from previous steps
    X, Y, P_x_given_omega1, P_x_given_omega2 = s2a()
    P_omega1 = 0.5
    P_omega2 = 0.5
    
    # Compute total probability P(x)
    P_x = P_omega1 * P_x_given_omega1 + P_omega2 * P_x_given_omega2
    
    # Compute posterior probabilities using Bayes theorem:
    # P(ω|x) = P(x|ω)P(ω) / P(x)
    P_omega1_given_x = (P_x_given_omega1 * P_omega1) / P_x
    P_omega2_given_x = (P_x_given_omega2 * P_omega2) / P_x
    
    # Plot posterior probabilities
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot P(ω1|x)
    contour1 = ax1.contourf(X, Y, P_omega1_given_x, levels=50, cmap='Reds')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('P(ω₁|x) - Posterior Class 1')
    ax1.plot(1, 1, 'ro', markersize=10)
    plt.colorbar(contour1, ax=ax1)
    
    # Plot P(ω2|x)
    contour2 = ax2.contourf(X, Y, P_omega2_given_x, levels=50, cmap='Blues')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('P(ω₂|x) - Posterior Class 2')
    ax2.plot(-1, -1, 'bo', markersize=10)
    plt.colorbar(contour2, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    # Compute Bayes error rate
    # P(error|x) = min(P(ω1|x), P(ω2|x))
    # P(error) = ∫ P(error|x) P(x) dx ≈ Σ min(P(ω1|x), P(ω2|x)) * P(x) * ΔxΔy
    
    P_error_given_x = np.minimum(P_omega1_given_x, P_omega2_given_x)
    
    # Since we're on a discrete grid, we approximate the integral by a sum
    # The area element ΔxΔy = 0.1 * 0.1 = 0.01
    delta_x = 0.1
    delta_y = 0.1
    area_element = delta_x * delta_y
    
    # Approximate the integral: Σ P(error|x) P(x) ΔxΔy
    Bayes_error = np.sum(P_error_given_x * P_x) * area_element
    
    print(f"Bayes Error Rate: {Bayes_error:.4f} ({Bayes_error*100:.2f}%)")
    
    # Plot the error regions
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, P_error_given_x, levels=50, cmap='hot')
    plt.colorbar(contour, label='P(error|x)')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Probability of Error P(error|x)')
    plt.plot(1, 1, 'ro', markersize=10, label='μ₁')
    plt.plot(-1, -1, 'bo', markersize=10, label='μ₂')
    plt.legend()
    
    plt.show()
    
    return P_omega1_given_x, P_omega2_given_x, Bayes_error

# Run Exercise 2
if __name__ == "__main__":
    print("Running Exercise 2...")
    X, Y, P1, P2 = s2a()
    X, Y, P_x = s2b()
    post1, post2, error = s2c()
