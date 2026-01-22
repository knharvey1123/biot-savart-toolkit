# Parametrized circle after one time step

import numpy as np
import matplotlib.pyplot as plt

# define planar circle
R = 1.0        # radius
N = 100        # number of points
theta = np.linspace(0, 2 * np.pi, N)

# circle time
gamma = np.stack((R * np.cos(theta), R * np.sin(theta), np.zeros_like(theta)), axis=1)
delta_gamma = np.stack((-R * np.sin(theta), R * np.cos(theta), np.zeros_like(theta)), axis=1)

#  curvature and torsion
curvature = np.full(N, 1 / R)
torsion = 0

# calculate integral result symbolically
# tangent, normal, and binormal vectors

# tangent = derivative of circle
tangent = delta_gamma

# normal = derivative of tangent, divided by curvature
normal = np.gradient(tangent, axis=0) / curvature[:, None]

# binormal = cross product of tangent and normal
binormal = np.cross(tangent, normal)

# apply at mesh points
# move mesh in time
# want to time evolve the mesh by gamma_t = kappa binormal
time_step = 0.1  # time step
gamma_new = gamma + time_step * (curvature[:, None] * binormal)

# plot original and new curve to visualize
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(gamma[:, 0], gamma[:, 1], gamma[:, 2], label='original circle')
ax.plot(gamma_new[:, 0], gamma_new[:, 1], gamma_new[:, 2], label='after time step')
ax.set_title('binormal flow: one time step')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.tight_layout()
plt.show()
