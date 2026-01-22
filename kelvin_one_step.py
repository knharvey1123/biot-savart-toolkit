import numpy as np
import matplotlib.pyplot as plt

# parameters
R = 1.0                      # radius
A = 0.01                     # amplitude
N = 10                       # number of wave cycles
num_points = 500             # number of points

s = np.linspace(0, 2 * np.pi, num_points)

# parametrization
x = R * np.cos(s) + A * np.cos(N * s) * np.cos(s)
y = R * np.sin(s) + A * np.cos(N * s) * np.sin(s)
z = -A * np.sin(N * s)

gamma = np.stack((x, y, z), axis=1)

# fairly certain this is where the problem is coming from
# but when i try to enforce periodicity with np.roll, the plot goes crazy
d_gamma = np.gradient(gamma, axis=0)
dd_gamma = np.gradient(d_gamma, axis=0)

# tangent vector
tangent = d_gamma
tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

# curvature
cross = np.cross(d_gamma, dd_gamma)
curvature = np.linalg.norm(cross, axis=1) / (np.linalg.norm(d_gamma, axis=1)**3)

# normal vector
normal = dd_gamma / curvature[:, None]
normal /= np.linalg.norm(normal, axis=1, keepdims=True)

# binormal vector
binormal = np.cross(tangent, normal)

# time step = 5 only for visualization purposes
time_step = 5
# evolved curve = gamma + kappa * binormal
gamma_new = gamma + time_step * (curvature[:, None] * binormal)

# visualize
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(gamma[:, 0], gamma[:, 1], gamma[:, 2], label='original curve')
ax.plot(gamma_new[:, 0], gamma_new[:, 1], gamma_new[:, 2], label='after time step')

ax.set_title('Binormal Flow (Kelvin Ring)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()
