'''
Time evolution of a Hasimoto soliton using BSI-induced flow
gamma_t = k * bsi_factor * B

status: need to verify correct
    - need to validate using arc length

problem: the stability checks are all being implemented, does this mean we're getting a simulation
not completely accurate to what we want?
 - code still runs without stability checks
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

import BSI_integral
import arcspline_interpolator


# non-periodic gradient function for solitons
def soliton_gradient(f, h):
    n = len(f)
    grad = np.zeros_like(f)

    # forward difference
    grad[0] = (f[1] - f[0]) / h

    # central difference
    for i in range(1, n-1):
        grad[i] = (f[i+1] - f[i-1]) / (2 * h)

    # backward difference
    grad[-1] = (f[-1] - f[-2]) / h

    return grad


# calculate curvature and binormal plane using frenet-serret framework
def compute_curvature_binormal(gamma, h):
    d_gamma = np.zeros_like(gamma)
    for j in range(3):
        d_gamma[:, j] = soliton_gradient(gamma[:, j], h)

    speed = np.linalg.norm(d_gamma, axis=1)
    speed = np.maximum(speed, 1e-10)
    tangent = d_gamma / speed[:, None]

    curvature_vector = np.zeros_like(tangent)
    for j in range(3):
        curvature_vector[:, j] = soliton_gradient(tangent[:, j], h)

    curvature = np.linalg.norm(curvature_vector, axis=1)

    normal = np.zeros_like(tangent)
    nonzero_curv = curvature > 1e-12
    normal[nonzero_curv] = curvature_vector[nonzero_curv] / curvature[nonzero_curv, None]

    binormal = np.cross(tangent, normal)
    return curvature, binormal, speed


def main():
    start_time = time.time()

    num_points = 300
    s = np.linspace(-50, 50, num_points)
    t = 0

    # params for soliton
    nu = 1.0
    tau_0 = 0.5

    eta = nu * (s - 2 * tau_0 * t)
    mu = 1 / (1 + tau_0**2)
    g = (2 * mu * (1 / np.cosh(eta))) / nu
    theta = tau_0 * s + (nu**2 - tau_0**2) * t

    # parametrization
    x = s - (2 * mu / nu) * np.tanh(eta)
    y = g * np.cos(theta)
    z = g * np.sin(theta)

    gamma = np.stack((x, y, z), axis=1)

    # time evolution params
    dt = 0.002
    n_steps = 700
    h = (s[-1] - s[0]) / (num_points - 1)  # non-periodic spacing
    phi = np.pi / 4

    '''need to figure out what this means / is doing / where the numbers are coming from'''
    # Reference radius for BSI calculation (characteristic scale of the soliton)
    R = 2.0 * mu / nu  # Characteristic radius based on soliton parameters

    # animation params
    animation_steps = 50
    step_interval = max(1, n_steps // animation_steps)

    print("Computing evolution states...")
    gamma_states = [gamma.copy()]
    gamma_current = gamma.copy()

    for step in range(n_steps):
        bsi_step = []

        for i in range(num_points):
            # handle boundary conditions
            if i == 0:
                # forward points at the start
                p1 = 2 * gamma_current[0] - gamma_current[1]
                p2 = gamma_current[0]
                p3 = gamma_current[1]
            elif i == num_points - 1:
                # backward points at the end
                p1 = gamma_current[i-1]
                p2 = gamma_current[i]
                p3 = 2 * gamma_current[i] - gamma_current[i-1]
            else:
                p1 = gamma_current[i-1]
                p2 = gamma_current[i]
                p3 = gamma_current[i+1]

            # get arc parameters using arcspline interpolation
            try:
                radius, length1, length2 = arcspline_interpolator.ArcSplineInterpolator.get_arc_parameters(p1, p2, p3)
            except (ValueError, TypeError):
                '''want to see if this code ever runs'''
                # fallback for problematic cases
                radius = 0.01
                # if radius == 0.01:
                #     print("i ran")
                length1 = np.linalg.norm(p2 - p1)
                length2 = np.linalg.norm(p3 - p2)

            eps = radius / R

            # compute bsi factors using BSI_integral
            bsi_val = BSI_integral.bsi_integral2(length1, length2, phi, eps, radius) * 10

            bsi_step.append(bsi_val)

        bsi_step = np.array(bsi_step)

        # curvature and binormal
        curvature, binormal, _ = compute_curvature_binormal(gamma_current, h)

        # evolve curve
        velocity = dt * curvature[:, None] * bsi_step[:, None] * binormal

        '''want to see if this code runs - it is'''
        # Limit velocity magnitude to prevent blow-up
        # velocity_magnitude = np.linalg.norm(velocity, axis=1)
        # max_velocity = 0.01
        # # if max_velocity == 0.01:
        # #     print('i also ran')
        # scale_factor = np.minimum(1.0, max_velocity / np.maximum(velocity_magnitude, 1e-10))
        # velocity *= scale_factor[:, None]

        gamma_current += velocity

        # Save states for animation
        if step % step_interval == 0:
            gamma_states.append(gamma_current.copy())

        if step % 50 == 0:
            print(f"Step {step}/{n_steps}, max BSI: {np.max(np.abs(bsi_step)):.6f}")

            # Check for numerical issues
            if np.any(np.isnan(gamma_current)) or np.any(np.isinf(gamma_current)):
                print("WARNING: NaN or Inf detected in simulation!")
                break

    print(f"Total states computed: {len(gamma_states)}")

    # Set up the figure
    fig = plt.figure(figsize=(12, 10))
    ax_xy = fig.add_subplot(221)                   # XY-plane
    ax_xz = fig.add_subplot(222)                   # XZ-plane
    ax_yz = fig.add_subplot(223)                   # YZ-plane
    ax_3d = fig.add_subplot(224, projection='3d')  # 3D plot

    # Compute plot limits
    all_x = np.concatenate([state[:, 0] for state in gamma_states])
    all_y = np.concatenate([state[:, 1] for state in gamma_states])
    all_z = np.concatenate([state[:, 2] for state in gamma_states])
    margin = 0.1
    xlim = (all_x.min() - margin, all_x.max() + margin)
    ylim = (all_y.min() - margin, all_y.max() + margin)
    zlim = (all_z.min() - margin, all_z.max() + margin)

    # Set limits & labels for each subplot
    ax_xy.set_xlim(xlim)
    ax_xy.set_ylim(ylim)
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    ax_xy.set_title('XY-plane')

    ax_xz.set_xlim(xlim)
    ax_xz.set_ylim(zlim)
    ax_xz.set_xlabel('X')
    ax_xz.set_ylabel('Z')
    ax_xz.set_title('XZ-plane')

    ax_yz.set_xlim(ylim)
    ax_yz.set_ylim(zlim)
    ax_yz.set_xlabel('Y')
    ax_yz.set_ylabel('Z')
    ax_yz.set_title('YZ-plane')

    ax_3d.set_xlim(xlim)
    ax_3d.set_ylim(ylim)
    ax_3d.set_zlim(zlim)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('3D view')

    # Initialize line objects
    line_xy, = ax_xy.plot([], [], 'b-', lw=1)
    line_xz, = ax_xz.plot([], [], 'g-', lw=1)
    line_yz, = ax_yz.plot([], [], 'r-', lw=1)
    line_3d, = ax_3d.plot([], [], [], 'k-', lw=2)

    # Shared time label for all subplots
    time_text = fig.text(0.02, 0.95, '', fontsize=12)

    def animate(frame):
        if frame < len(gamma_states):
            gamma_frame = gamma_states[frame]

            # No need to close the loop for solitons - they're not periodic
            x_vals = gamma_frame[:, 0]
            y_vals = gamma_frame[:, 1]
            z_vals = gamma_frame[:, 2]

            # Update 2D projections
            line_xy.set_data(x_vals, y_vals)
            line_xz.set_data(x_vals, z_vals)
            line_yz.set_data(y_vals, z_vals)

            # Update 3D plot
            line_3d.set_data_3d(x_vals, y_vals, z_vals)

            # Update time label
            current_time = frame * step_interval * dt
            time_text.set_text(f'Time: {current_time:.5f}')

        return line_xy, line_xz, line_yz, line_3d, time_text

    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(
        fig, animate, frames=len(gamma_states),
        blit=False, repeat=True
    )
    _ = anim

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    plt.tight_layout()
    plt.show()

    print(f"Animation complete! Total evolution time: {n_steps * dt:.5f}")


if __name__ == "__main__":
    main()
