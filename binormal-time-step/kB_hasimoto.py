'''
Time evolution of a Hasimoto soliton
Using the Local Induction Approximation

Note: strange oscillations, potential computer precision
'''

import numpy as np
import matplotlib.pyplot as plt

s = np.linspace(-200, 200, 1500)
t = 0

# parameters
nu = 1.0
tau_0 = 0.5

eta = nu * (s - 2 * tau_0 * t)
mu = mu = 1 / (1 + tau_0**2)
gamma = (2 * mu * (1 / np.cosh(eta))) / nu
theta = tau_0 * s + (nu**2 - tau_0**2) * t

# x, y, z
x = s - (2 * mu / nu) * np.tanh(eta)
y = gamma * np.cos(theta)
z = gamma * np.sin(theta)

soliton = np.stack((x, y, z), axis=1)

# time step
dt = 0.01
n_steps = 10001

# 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for step in range(n_steps):
    # finite difference derivative of gamma
    tangent = np.gradient(soliton, axis=0, edge_order=2)

    # normalize tangent
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent_unit = tangent / tangent_norm

    # second derivative
    tangent2 = np.gradient(tangent_unit, axis=0, edge_order=2)
    curvature_vec = tangent2
    curvature_mag = np.linalg.norm(curvature_vec, axis=1, keepdims=True)

    # normal
    normal = curvature_vec / (curvature_mag + 1e-8)

    # binormal
    binormal = np.cross(tangent_unit, normal)

    # do the time step
    soliton += dt * curvature_mag * binormal

    # visualize time steps
    if step % 1000 == 0:
        ax.plot(soliton[:, 0], soliton[:, 1], soliton[:, 2], label=f"t = {step * dt:.2f}")

# plot
ax.set_title('Time Evolution - Hasimoto soliton')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.tight_layout()
plt.show()
