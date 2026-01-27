'''
Hasimoto soliton after one time step
Computed and visualized using the Local Induction Approximation and an Euler method
'''

import numpy as np
import matplotlib.pyplot as plt

s = np.linspace(-200, 200, 3000)
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

# time step it
# first and second derivatives
d_soliton = np.gradient(soliton, axis=0)
dd_soliton = np.gradient(d_soliton, axis=0)

# tangent vector
tangent = d_soliton
tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

# curvature
cross = np.cross(d_soliton, dd_soliton)
curvature = np.linalg.norm(cross, axis=1) / (np.linalg.norm(d_soliton, axis=1)**3)

# normal vector
normal = dd_soliton
normal /= np.linalg.norm(normal, axis=1, keepdims=True)

# binormal vector
binormal = np.cross(tangent, normal)

time_step = 10.0
soliton_new = soliton + time_step * (curvature[:, None] * binormal)

# visualize
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(soliton[:, 0], soliton[:, 1], soliton[:, 2], '-', label='original curve')
ax.plot(soliton_new[:, 0], soliton_new[:, 1], soliton_new[:, 2], '--', label='after time step')

ax.set_title('Binormal Flow (Hasimoto soliton)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()
