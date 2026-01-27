import numpy as np
from scipy import integrate


def bsi_integral(L1, L2, phi, epsilon):

    c1 = epsilon**2 - 2 * epsilon*np.cos(phi) + 2
    c2 = 2 * epsilon * np.cos(phi) - 2

    def integrand(x):
        return (np.cos(x) - 1) / (c1 + c2 * np.cos(x))**1.5

    result, _ = integrate.quad(integrand, L1, L2)
    return result


def bsi_integral2(L1, L2, phi, epsilon, radius):
    # Convert arc lengths to angles
    theta1 = L1 / radius if radius > 0 else 0
    theta2 = L2 / radius if radius > 0 else 0

    c1 = epsilon**2 - 2 * epsilon*np.cos(phi) + 2
    c2 = 2 * epsilon * np.cos(phi) - 2

    def integrand(x):
        return (np.cos(x) - 1) / (c1 + c2 * np.cos(x))**1.5

    result, _ = integrate.quad(integrand, theta1, theta2)
    return result
