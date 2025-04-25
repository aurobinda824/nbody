import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import openpyxl
from scipy.integrate import quad, simpson
from scipy.integrate import odeint
from matplotlib.colors import PowerNorm

class ParticleMeshSimulation:
    def __init__(self, grid_size, num_particles):
        self.grid_size = grid_size
        self.num_particles = num_particles

        self.positions = cp.zeros((self.num_particles, 3))
        self.velocities = cp.zeros((self.num_particles, 3))

        self.density_grid = cp.zeros((grid_size + 1, grid_size + 1, grid_size + 1))
        self.potential_grid = cp.zeros_like(self.density_grid)
        self.force_grid = cp.zeros((3, grid_size + 1, grid_size + 1, grid_size + 1))

        self.a = 0.01
        self.omega_m = 0.315
        self.omega_b = 0.0493
        self.omega_lambda = 0.685
        self.omega_k = 0.0
        self.omega_r = 9.24e-5  
        self.h = 0.674
        self.ns = 0.965
        self.sigma8 = 0.811
        self.lna = cp.log(self.a)
        self.box_size = 1.0

    def initial_positions(self):
        q = cp.linspace(0, self.grid_size, self.grid_size, endpoint=False)
        qx, qy, qz = cp.meshgrid(q, q, q, indexing='ij')
        self.positions = cp.column_stack((qx.ravel(), qy.ravel(), qz.ravel()))
        self.positions %= self.grid_size

    def hubble(self, a=None):
        a = a if a is not None else self.a
        return self.h * 100 * self.E(a)

    def E(self, a=None):
        a = a if a is not None else self.a
        return cp.sqrt(self.omega_r * a**-4 + self.omega_m * a**-3 +
                       self.omega_k * a**-2 + self.omega_lambda)

    def growth_factor(self, a=None):
        a = a if a is not None else self.a
        def growth_equation(D, lna, omega_m, omega_r, omega_k, omega_lambda):
            a = np.exp(lna)
            E2 = omega_r * a**-4 + omega_m * a**-3 + omega_k * a**-2 + omega_lambda
            Omega_m_a = omega_m * a**-3 / E2
            dH_over_H2 = -0.5 * (4 * omega_r * a**-4 + 3 * omega_m * a**-3 + 2 * omega_k * a**-2) / E2
            dD_dlna = D[1]
            d2D_dlna2 = 1.5 * Omega_m_a * D[0] - (2 + dH_over_H2) * D[1]
            return [dD_dlna, d2D_dlna2]
        omega_m = self.omega_m
        omega_r = self.omega_r
        omega_k = self.omega_k
        omega_lambda = self.omega_lambda
        a_init = 1e-3
        lna = np.linspace(np.log(a_init), 0, 100)
        a_vals = np.exp(lna)
        D0 = a_init
        dD_dlna0 = D0
        initial_conditions = [D0, dD_dlna0]
        sol = odeint(growth_equation, initial_conditions, lna, 
                     args=(omega_m, omega_r, omega_k, omega_lambda))
        D = sol[:, 0]
        D /= D[-1]
        return np.interp(np.log(a), lna, D)

    def growth_rate(self, a=None):
        a = a if a is not None else self.a
        delta_a = 1e-6
        D1 = self.growth_factor(a)
        D2 = self.growth_factor(a + delta_a)
        return np.log(D2 / D1) / np.log((a + delta_a) / a)


    def transfer_function(self, k):
        gamma = self.omega_m * self.h * np.exp(-self.omega_b * (1 + np.sqrt(2 * self.h) / self.omega_m))
        q = k / (gamma * self.h)
        L0 = np.log(1 + 2.34 * q)
        C0 = 1 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4
        return L0 / (2.34 * q * C0**0.25)
    
    def window_function(self, k, R=8):
        x = k * R
        return cp.abs(3 * (np.sin(x) - x * np.cos(x)) / x**3)
    
    def power_spectrum(self, ks):
        def integrand(k):
            T = self.transfer_function(k)
            W = self.window_function(k)
            return k**(self.ns + 2) * T**2 * W**2
        
        k_min, k_max = 1e-4, 1e2  
        I, _ = quad(integrand, k_min, k_max)
        A = self.sigma8**2 / I
        
        T = cp.array([self.transfer_function(k) for k in ks])
        return A * T**2 * ks**self.ns

    def zeldovich(self, delta_a=0.01):
        k_freq = 2 * np.pi * cp.fft.fftfreq(self.grid_size, d=1/self.grid_size)
        kx, ky, kz = cp.meshgrid(k_freq, k_freq, k_freq, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
        k_mag[0,0,0] = 1.0

        rand_real = cp.random.normal(0, 1, (self.grid_size, self.grid_size, self.grid_size))
        rand_imag = cp.random.normal(0, 1, (self.grid_size, self.grid_size, self.grid_size))
        delta_k = (rand_real + 1j * rand_imag)
        delta_k = 0.5 * (delta_k + cp.conj(delta_k[::-1, ::-1, ::-1]))
        
        P_k = self.power_spectrum(k_mag)
        delta_k *= cp.sqrt(P_k / 2) * self.grid_size ** 3
        
        phi_k = -delta_k / (k_mag**2)
        phi_k[0,0,0] = 0
        psi_x = cp.fft.ifftn(-1j * kx * phi_k).real
        psi_y = cp.fft.ifftn(-1j * ky * phi_k).real
        psi_z = cp.fft.ifftn(-1j * kz * phi_k).real

        D = self.growth_factor()
        f = self.growth_rate()
        Hz = self.hubble()

        displacements = D * cp.column_stack([psi_x.ravel(), psi_y.ravel(), psi_z.ravel()])
        self.positions += displacements
        self.positions %= self.grid_size

        self.velocities = -(f * Hz) * displacements
        
        print("Zel'dovich IC generated with:")
        print(f"- Box size: {self.box_size:.1f} Mpc/h")
        print(f"- Grid size: {self.grid_size}^3")
        print(f"- Max displacement: {cp.max(cp.abs(displacements)):.3f} Mpc/h")
        print(f"- Max velocity: {cp.max(cp.abs(self.velocities)):.3f} km/s")


    def mass_assign(self):
        self.density_grid = cp.zeros((self.grid_size + 1, self.grid_size + 1, self.grid_size + 1))
        ijk = cp.floor(self.positions).astype(cp.int32)
        dxdyz = self.positions - ijk
        dx, dy, dz = dxdyz[:, 0], dxdyz[:, 1], dxdyz[:, 2]
        i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
        ip1, jp1, kp1 = i + 1, j + 1, k + 1
        w000 = (1 - dx) * (1 - dy) * (1 - dz)
        w100 = dx * (1 - dy) * (1 - dz)
        w010 = (1 - dx) * dy * (1 - dz)
        w001 = (1 - dx) * (1 - dy) * dz
        w110 = dx * dy * (1 - dz)
        w101 = dx * (1 - dy) * dz
        w011 = (1 - dx) * dy * dz
        w111 = dx * dy * dz
        inds = cp.stack([
            cp.stack([i, j, k], axis=1),
            cp.stack([ip1, j, k], axis=1),
            cp.stack([i, jp1, k], axis=1),
            cp.stack([i, j, kp1], axis=1),
            cp.stack([ip1, jp1, k], axis=1),
            cp.stack([ip1, j, kp1], axis=1),
            cp.stack([i, jp1, kp1], axis=1),
            cp.stack([ip1, jp1, kp1], axis=1),
        ], axis=1)

        weights = cp.stack([w000, w100, w010, w001, w110, w101, w011, w111], axis=1)

        inds = inds % (self.grid_size + 1)
        flat_inds = inds.reshape(-1, 3)
        flat_weights = weights.ravel()

        linear_inds = (
            flat_inds[:, 0] * (self.grid_size + 1)**2 +
            flat_inds[:, 1] * (self.grid_size + 1) +
            flat_inds[:, 2]
        )

        flat_grid = self.density_grid.ravel()
        cp.add.at(flat_grid, linear_inds, flat_weights)
        self.density_grid = flat_grid.reshape(self.density_grid.shape)
        self.density_grid -= 1

    def solve_poisson(self):
        kx, ky, kz = cp.meshgrid(
            cp.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            cp.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            cp.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            indexing='ij'
        )

        k2 = kx ** 2 + ky ** 2 + kz ** 2
        k2[0, 0, 0] = 1
        greens_function = -1 / k2
        greens_function[0, 0, 0] = 0

        self.potential_grid = cp.fft.ifftn(cp.fft.fftn(self.density_grid) * greens_function).real

    def compute_forces(self):
        self.force_grid = cp.zeros((3, self.grid_size + 1, self.grid_size + 1, self.grid_size + 1))
        
        self.force_grid[0, 1:-1, :, :] = -(self.potential_grid[2:, :, :] - self.potential_grid[:-2, :, :]) / 2
        self.force_grid[1, :, 1:-1, :] = -(self.potential_grid[:, 2:, :] - self.potential_grid[:, :-2, :]) / 2
        self.force_grid[2, :, :, 1:-1] = -(self.potential_grid[:, :, 2:] - self.potential_grid[:, :, :-2]) / 2
        
        self.force_grid[0, 0, :, :] = -(self.potential_grid[1, :, :] - self.potential_grid[-2, :, :]) / 2
        self.force_grid[0, -1, :, :] = self.force_grid[0, 0, :, :]
        self.force_grid[1, :, 0, :] = -(self.potential_grid[:, 1, :] - self.potential_grid[:, -2, :]) / 2
        self.force_grid[1, :, -1, :] = self.force_grid[1, :, 0, :]
        self.force_grid[2, :, :, 0] = -(self.potential_grid[:, :, 1] - self.potential_grid[:, :, -2]) / 2
        self.force_grid[2, :, :, -1] = self.force_grid[2, :, :, 0]

    def update_velocity(self, delta_ln_a):
        const = 1.5 * self.hubble() * 100 / (self.a * self.E(self.a)) * (delta_ln_a / 2) * np.exp(self.lna)
        pos_floor = cp.floor(self.positions).astype(cp.int32)
        dxdyz = self.positions - pos_floor

        dx, dy, dz = dxdyz[:, 0], dxdyz[:, 1], dxdyz[:, 2]
        i, j, k = pos_floor[:, 0], pos_floor[:, 1], pos_floor[:, 2]
        ip1, jp1, kp1 = i + 1, j + 1, k + 1

        w000 = (1 - dx) * (1 - dy) * (1 - dz)
        w100 = dx * (1 - dy) * (1 - dz)
        w010 = (1 - dx) * dy * (1 - dz)
        w001 = (1 - dx) * (1 - dy) * dz
        w110 = dx * dy * (1 - dz)
        w101 = dx * (1 - dy) * dz
        w011 = (1 - dx) * dy * dz
        w111 = dx * dy * dz
        weights = cp.stack([w000, w100, w010, w001, w110, w101, w011, w111], axis=1)
        indices = cp.stack([
            cp.stack([i, j, k], axis=1),
            cp.stack([ip1, j, k], axis=1),
            cp.stack([i, jp1, k], axis=1),
            cp.stack([i, j, kp1], axis=1),
            cp.stack([ip1, jp1, k], axis=1),
            cp.stack([ip1, j, kp1], axis=1),
            cp.stack([i, jp1, kp1], axis=1),
            cp.stack([ip1, jp1, kp1], axis=1)
        ], axis=1)
        force_contributions = cp.zeros((3, self.num_particles, 8), dtype=cp.float32)
        for axis in range(3):
            grid_vals = self.force_grid[axis]
            force_contributions[axis] = grid_vals[indices[:, :, 0], indices[:, :, 1], indices[:, :, 2]]
        force_interp = cp.sum(force_contributions * weights[None, :, :], axis=2)
        self.velocities += (force_interp.T * const)


    def leapfrog(self, delta_ln_a):
        self.update_velocity(delta_ln_a)
        self.positions += self.velocities * delta_ln_a / (self.a**2 * self.E(self.a)) * np.exp(self.lna)
        self.positions %= self.grid_size
        self.lna += delta_ln_a
        self.a = cp.exp(self.lna)
        self.update_velocity(delta_ln_a)

    def kinetic(self):
        return 0.5 * np.sum(self.velocities**2)

    def potential(self):
        return 0.5 * np.sum(self.density_grid * self.potential_grid)

    def visualize(self, bins=512, name='sim'):
        x = cp.asnumpy(self.positions[:, 0])
        y = cp.asnumpy(self.positions[:, 1])
        heatmap, xedges, yedges = np.histogram2d(
            x, y, bins=bins, range=[[0, self.grid_size], [0, self.grid_size]]
        )
        plt.figure(figsize=(8, 8))
        plt.imshow(
            heatmap.T,
            origin='lower',
            extent=[0, self.grid_size, 0, self.grid_size],
            cmap='inferno',
            norm=PowerNorm(gamma=0.4)
        )
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{name}.png')
        plt.close()


    def visualize_potential(self, slice_index=None, projection='2d'):
        if slice_index is None:
            slice_index = self.grid_size // 2

        plt.figure(figsize=(8, 6))
        plt.imshow(self.potential_grid[slice_index, :, :], cmap='viridis', origin='lower',
                   extent=[0, 1, 0, 1], aspect='auto')
        plt.colorbar(label='Potential')
        plt.title(f'Potential Field (Slice at index {slice_index})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def compute_power_spectrum(self, delta_x, n_bins=20):
        N = delta_x.shape[0]
        delta_k = np.fft.fftn(delta_x)
        kx = np.fft.fftfreq(N, d=self.grid_size/N) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=self.grid_size/N) * 2 * np.pi
        kz = np.fft.fftfreq(N, d=self.grid_size/N) * 2 * np.pi
        k_grid = np.sqrt(kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2)
        
        k_max = np.max(k_grid)
        k_bins = np.logspace(np.log10(2 * np.pi / self.grid_size), np.log10(k_max), n_bins)
        power = np.histogram(k_grid, bins=k_bins, weights=np.abs(delta_k)**2)[0]
        counts = np.histogram(k_grid, bins=k_bins)[0]

        mask = counts > 0
        k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
        P_k = (self.grid_size ** 3 / N ** 6) * (power[mask] / counts[mask])
        
        return k_centers[mask], P_k


    def solve_telegraph(self, dp, kappa=2.5, c_g=1.0, max_iter=100):
        psi = cp.zeros_like(self.potential_grid)
        H = self.hubble()
        dHdp = -1 * (self.hubble(a=self.a) - self.hubble(a=self.a + cp.exp(self.lna) * dp)) / dp
        laplacian_coeff = 1 / (self.a * H * self.grid_size) ** 2
        damping_coeff = (1/self.a + dHdp * cp.exp(-self.lna) / H + 2 * kappa / (self.a * H))

        exp_lna = cp.exp(self.lna)

        for iteration in range(max_iter):
            self.potential_grid += dp * exp_lna * psi

            # Use CuPy's array shifting (cp.roll) to compute Laplacian
            phi_xp = cp.roll(self.potential_grid, -1, axis=0)
            phi_xm = cp.roll(self.potential_grid, 1, axis=0)
            phi_yp = cp.roll(self.potential_grid, -1, axis=1)
            phi_ym = cp.roll(self.potential_grid, 1, axis=1)
            phi_zp = cp.roll(self.potential_grid, -1, axis=2)
            phi_zm = cp.roll(self.potential_grid, 1, axis=2)

            laplacian = (phi_xp + phi_xm + phi_yp + phi_ym + phi_zp + phi_zm - 6 * self.potential_grid)

            # Update psi using CuPy operations
            psi += dp * exp_lna * (-damping_coeff * psi + laplacian_coeff * (laplacian - self.density_grid))

        return iteration + 1

   