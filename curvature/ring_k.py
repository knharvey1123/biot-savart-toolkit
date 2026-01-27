'''
Curvature plot for a ring
'''

import numpy as np
import matplotlib.pyplot as plt

R = 4.0
N = 100
theta = np.linspace(0, 2 * np.pi, N)

gamma = np.stack((R * np.cos(theta), R * np.sin(theta), np.zeros_like(theta)), axis=1)

# evolution parameters
time_step = 100
num_steps = 5

plt.figure(figsize=(10, 5))

for step in range(num_steps + 1):
    # derivatives
    d_gamma = np.stack((-R * np.sin(theta), R * np.cos(theta), np.zeros_like(theta)), axis=1)
    dd_gamma = np.stack((-R * np.cos(theta), -R * np.sin(theta), np.zeros_like(theta)), axis=1)
    cross = np.cross(d_gamma, dd_gamma)

    # curvature
    curvature = np.linalg.norm(cross, axis=1) / (np.linalg.norm(d_gamma, axis=1))**3

    # plot curvature
    label = f"step {step}" if step > 0 else "initial"
    plt.plot(theta, curvature, label=label)

    # evolve curve
    gamma = gamma + time_step * (curvature[:, None] * cross)

# plot curvature
plt.title("Curvature (circle)")
plt.xlabel("θ")
plt.ylabel("κ(θ)")
plt.grid(True)
plt.show()
