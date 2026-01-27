'''
Time evolution of a Kelvin Ring
Using the Local Induction Approximation
'''

import numpy as np
import matplotlib.pyplot as plt

# parameters
R = 1.0                      # radius
A = 0.5                      # amplitude
N = 10                       # number of wave cycles
num_points = 500             # number of points

# parameterize
phi = np.linspace(0, 2 * np.pi, num_points)

# define kelvin ring
x = R * np.cos(phi) + A * np.cos(N * phi) * np.cos(phi)
y = R * np.sin(phi) + A * np.cos(N * phi) * np.sin(phi)
z = -A * np.sin(N * phi)
gamma = np.stack((x, y, z), axis=1)

# time step
dt = 0.01
n_steps = 201
my_array = [i for i in range(0, 201)]

# 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for step in range(n_steps):
    # finite difference derivative of gamma
    tangent = np.gradient(gamma, axis=0, edge_order=2)

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
    gamma += dt * curvature_mag * binormal

    # visualize time steps
    if step % 50 == 0 or step == n_steps - 1:
        ax.plot(gamma[:, 0], gamma[:, 1], gamma[:, 2], label=f"t = {step * dt:.2f}")

        segment_lengths = np.linalg.norm(np.roll(gamma, -1, axis=0) - gamma, axis=1)
        total_length = np.sum(segment_lengths)
        print(f"step {step}: length = {total_length:.6f}")

# plot
ax.set_title('Time Evolution - Kelvin ring')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.tight_layout()
plt.show()
