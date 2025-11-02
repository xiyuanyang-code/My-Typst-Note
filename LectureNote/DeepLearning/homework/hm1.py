import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def volume_of_d_ball(r, d):
    """
    Calculate the volume V_d(r) of a d-dimensional ball with radius r.
    
    Formula: V_d(r) = [pi^(d/2) / Gamma(d/2 + 1)] * r^d
    
    Parameters:
    r (float or array-like): Radius of the ball.
    d (int): Dimension.
    
    Returns:
    float or array-like: Volume of the d-dimensional ball.
    """
    # Calculate volume constant C_d = pi^(d/2) / Gamma(d/2 + 1)
    C_d = (np.pi**(d / 2)) / gamma(d / 2 + 1)
    
    # Volume V_d(r) = C_d * r^d
    V_d_r = C_d * (r**d)
    
    return V_d_r

def plot_d_ball_volume_curves():
    """
    Plot the volume curves of balls in dimensions d=5, 10, 20 as a function of radius.
    """
    # 1. Define dimensions to plot
    dimensions = [5, 10, 20]
    
    # 2. Define radius range (from 0 to 1.5 to better observe growth trends)
    r_values = np.linspace(0, 1, 300)
    
    # 3. Initialize plotting
    plt.figure(figsize=(10, 6))
    plt.title('Volume $V_d(r)$ of High-Dimensional Balls vs Radius $r$', fontsize=16)
    plt.xlabel('Radius $r$', fontsize=14)
    plt.ylabel('Volume $V_d(r)$', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 4. Loop to calculate and plot each curve
    for d in dimensions:
        # Calculate volume values for the corresponding dimension
        V_d_values = volume_of_d_ball(r_values, d)
        
        # Plot the curve with label
        plt.plot(r_values, V_d_values, 
                 label=f'$d = {d}$', 
                 linewidth=2.5)

    # 5. Add annotations and legend
    
    # Mark unit radius (r=1) position
    plt.axvline(x=1.0, color='gray', linestyle=':', linewidth=1.5)
    plt.text(1.02, plt.ylim()[1] * 0.9, '$r=1$ (unit radius)', 
             rotation=0, color='gray')
    
    plt.legend(fontsize=12, loc='upper left')
    plt.ylim(bottom=0) # Ensure volume starts from 0
    plt.savefig("./LectureNote/DeepLearning/homework/d_ball.pdf")

# Execute visualization function
plot_d_ball_volume_curves()