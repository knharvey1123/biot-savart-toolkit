'''
Curvature plot for Kelvin ring
- parametrizes a kelvin ring
- calculates tangent, normal, binormal
- calculates curvature and plots in 2D plot
- evolves the kelvin ring in time using a kappa*binormal
'''

import numpy as np
import matplotlib.pyplot as plt

# parameters
R = 1.0
A = 0.001
N = 10
num_points = 500
phi = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

# kelvin ring
x = R * np.cos(phi) + A * np.cos(N * phi) * np.cos(phi)
y = R * np.sin(phi) + A * np.cos(N * phi) * np.sin(phi)
z = -A * np.sin(N * phi)
gamma = np.stack((x, y, z), axis=1)

# evolution parameters
time_step = 100
num_steps = 5

plt.figure(figsize=(10, 5))

for step in range(num_steps + 1):
    # derivatives
    d_gamma = np.gradient(gamma, axis=0, edge_order=2)
    dd_gamma = np.gradient(d_gamma, axis=0, edge_order=2)
    cross = np.cross(d_gamma, dd_gamma)

    # curvature
    curvature = np.linalg.norm(cross, axis=1) / (np.linalg.norm(d_gamma, axis=1))**3

    # plot curvature
    label = f"step {step}" if step > 0 else "initial"
    plt.plot(phi, curvature, label=label)

    # evolve curve
    gamma = gamma + time_step * (curvature[:, None] * cross)

# plot
plt.title("Curvature Evolution (Kelvin ring)")
plt.xlabel("φ")
plt.ylabel("κ(φ)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
