'''
Curvature plot for Hasimoto soliton
- parametrizes a Hasimoto soliton
- calculates tangent, normal, binormal
- calculates curvature and plots in 2D plot
- evolves the Hasimoto soliton in time using a kappa*binormal
'''

import numpy as np
import matplotlib.pyplot as plt

# parameters
s = np.linspace(-100, 100, 1500)
t = 0

nu = 1.0
tau_0 = 0.5

eta = nu * (s - 2 * tau_0 * t)
mu = mu = 1 / (1 + tau_0**2)
gamma = (2 * mu * (1 / np.cosh(eta))) / nu
theta = tau_0 * s + (nu**2 - tau_0**2) * t

# parameterize
x = s - (2 * mu / nu) * np.tanh(eta)
y = gamma * np.cos(theta)
z = -gamma * np.sin(theta)

soliton = np.stack((x, y, z), axis=1)

# evolution parameters
time_step = 0.01
num_steps = 100

plt.figure(figsize=(10, 5))

for step in range(num_steps + 1):
    # derivatives
    d_soliton = np.gradient(soliton, axis=0, edge_order=2)
    dd_soliton = np.gradient(d_soliton, axis=0, edge_order=2)
    cross = np.cross(d_soliton, dd_soliton)

    # curvature
    curvature = np.linalg.norm(cross, axis=1) / (np.linalg.norm(d_soliton, axis=1))**3

    # plot curvature
    if step % 10 == 0:
        label = f"step {step}" if step % 10 == 0 else "initial"
        plt.plot(s, curvature, label=label)

    # evolve curve
    soliton = soliton + time_step * (curvature[:, None] * cross)

# plot
plt.title("Curvature Evolution (Hasimoto soliton)")
plt.xlabel("s")
plt.ylabel("κ(s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
