import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import openpyxl
from scipy.integrate import quad, simpson
from matplotlib.colors import PowerNorm

class ParticleMeshSimulation:
    def __init__(self, grid_size, num_particles):
        self.grid_size = grid_size
        self.num_particles = num_particles

        self.positions = np.zeros((self.num_particles, 3))
        self.velocities = np.zeros((self.num_particles, 3))

        self.density_grid = np.zeros((grid_size + 1, grid_size + 1, grid_size + 1))
        self.potential_grid = np.zeros_like(self.density_grid)
        self.force_grid = np.zeros((3, grid_size + 1, grid_size + 1, grid_size + 1))

        self.a = 0.01
        self.omega_m = 1
        self.omega_b = 0.1156
        self.omega_lambda = 0
        self.omega_k = 0.0
        self.omega_r = 0.0
        self.h = 0.674
        self.ns = 0.965
        self.sigma8 = 0.811
        self.lna = np.log(self.a)
        self.box_size = 1.0

    def initial_positions(self):
        q = np.linspace(0, self.grid_size, self.grid_size, endpoint=False) + 0.5
        qx, qy, qz = np.meshgrid(q, q, q, indexing='ij')
        self.positions = np.column_stack((qx.ravel(), qy.ravel(), qz.ravel()))
        self.positions %= self.grid_size

    def hubble(self, a=None):
        a = a if a is not None else self.a
        return self.h * 100 * self.E(a)

    def E(self, a=None):
        a = a if a is not None else self.a
        return np.sqrt(self.omega_r * a**-4 + self.omega_m * a**-3 +
                       self.omega_k * a**-2 + self.omega_lambda)

    def growth_factor(self, a=None):
        return a if a is not None else self.a


    def growth_rate(self, a=None):
        a = a if a is not None else self.a
        return (self.omega_m * a**-3 / self.E(a)**2)**0.55

    def transfer_function(self, k):
        gamma = self.omega_m * self.h * np.exp(-self.omega_b * (1 + np.sqrt(2 * self.h) / self.omega_m))
        q = k / (gamma * self.h)
        L0 = np.log(1 + 2.34 * q)
        C0 = 1 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4
        return L0 / (2.34 * q * C0**0.25)
    
    def window_function(self, k, R=8):
        x = k * R
        return np.abs(3 * (np.sin(x) - x * np.cos(x)) / x**3)
    
    def power_spectrum(self, ks):
        def integrand(k):
            T = self.transfer_function(k)
            W = self.window_function(k)
            return k**(self.ns + 2) * T**2 * W**2
        
        k_min, k_max = 1e-4, 1e2  
        I, _ = quad(integrand, k_min, k_max)
        A = self.sigma8**2 / I
        
        T = np.array([self.transfer_function(k) for k in ks])
        return A * T**2 * ks**self.ns * self.a

    def zeldovich(self, delta_a=0.01):
        k_freq = 2 * np.pi * np.fft.fftfreq(self.grid_size, d=1/self.grid_size)
        kx, ky, kz = np.meshgrid(k_freq, k_freq, k_freq, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
        k_mag[0,0,0] = 1.0

        rand_real = np.random.normal(0, 1/np.sqrt(2), (self.grid_size, self.grid_size, self.grid_size))
        rand_imag = np.random.normal(0, 1/np.sqrt(2), (self.grid_size, self.grid_size, self.grid_size))
        delta_k = (rand_real + 1j * rand_imag)

        # Enforce Hermitian symmetry (to ensure real-valued field in real space)
        delta_k = 0.5 * (delta_k + np.conj(delta_k[::-1, ::-1, ::-1]))
        
        P_k = self.power_spectrum(k_mag)
        delta_k *= np.sqrt(P_k / 2) * self.grid_size ** 3
        
        phi_k = -delta_k / (k_mag**2)
        phi_k[0,0,0] = 0
        psi_x = np.fft.ifftn(1j * kx * phi_k).real
        psi_y = np.fft.ifftn(1j * ky * phi_k).real
        psi_z = np.fft.ifftn(1j * kz * phi_k).real

        D = self.growth_factor()
        f = self.growth_rate()
        Hz = self.hubble()

        displacements = D * np.column_stack([psi_x.ravel(), psi_y.ravel(), psi_z.ravel()])
        self.positions -= displacements
        self.positions %= self.grid_size

        self.velocities = -(f * Hz) * displacements
        
        print("Zel'dovich IC generated with:")
        print(f"- Box size: {self.box_size:.1f} Mpc/h")
        print(f"- Grid size: {self.grid_size}^3")
        print(f"- Max displacement: {np.max(np.abs(displacements)):.3f} Mpc/h")
        print(f"- Max velocity: {np.max(np.abs(self.velocities)):.3f} km/s")


    def mass_assign(self, scheme):
        self.density_grid = np.zeros((self.grid_size + 1, self.grid_size + 1, self.grid_size + 1))
        if scheme == 'CIC':
            for pos in self.positions:
                i, j, k = np.floor(pos).astype(int)

                dx, dy, dz = pos - np.array([i, j, k])
                ip1, jp1, kp1 = i + 1, j + 1, k + 1
                w000 = (1 - dx) * (1 - dy) * (1 - dz)
                w100 = dx * (1 - dy) * (1 - dz)
                w010 = (1 - dx) * dy * (1 - dz)
                w001 = (1 - dx) * (1 - dy) * dz
                w110 = dx * dy * (1 - dz)
                w101 = dx * (1 - dy) * dz
                w011 = (1 - dx) * dy * dz
                w111 = dx * dy * dz

                self.density_grid[i, j, k] += w000
                self.density_grid[ip1, j, k] += w100
                self.density_grid[i, jp1, k] += w010
                self.density_grid[i, j, kp1] += w001
                self.density_grid[ip1, jp1, k] += w110
                self.density_grid[ip1, j, kp1] += w101
                self.density_grid[i, jp1, kp1] += w011
                self.density_grid[ip1, jp1, kp1] += w111

        if scheme == 'NGP':  
            for pos in self.positions:
                i, j, k = np.floor(pos + 0.5).astype(int)
                i %= self.grid_size
                j %= self.grid_size
                k %= self.grid_size
                self.density_grid[i, j, k] += 1

        if scheme == 'TSC':
            for pos in self.positions:
                i, j, k = np.floor(pos).astype(int) % self.grid_size

                dx, dy, dz = pos - np.array([i, j, k])

                im1, jm1, km1 = (i - 1) % self.grid_size, (j - 1) % self.grid_size, (k - 1) % self.grid_size
                ip1, jp1, kp1 = (i + 1) % self.grid_size, (j + 1) % self.grid_size, (k + 1) % self.grid_size

                wx = np.array([0.5 * (1 - dx) ** 2, 0.75 - (dx - 0.5) ** 2, 0.5 * dx ** 2])
                wy = np.array([0.5 * (1 - dy) ** 2, 0.75 - (dy - 0.5) ** 2, 0.5 * dy ** 2])
                wz = np.array([0.5 * (1 - dz) ** 2, 0.75 - (dz - 0.5) ** 2, 0.5 * dz ** 2])

                grid_indices_x = np.array([im1, i, ip1])
                grid_indices_y = np.array([jm1, j, jp1])
                grid_indices_z = np.array([km1, k, kp1])

                for xi, wxi in zip(grid_indices_x, wx):
                    for yi, wyi in zip(grid_indices_y, wy):
                        for zi, wzi in zip(grid_indices_z, wz):
                            self.density_grid[xi, yi, zi] += wxi * wyi * wzi

        self.density_grid -= 1

    def solve_poisson(self):
        kx, ky, kz = np.meshgrid(
            np.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            np.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            np.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            indexing='ij'
        )

        k2 = kx ** 2 + ky ** 2 + kz ** 2
        k2[0, 0, 0] = 1
        greens_function = -1 / k2
        greens_function[0, 0, 0] = 0

        self.potential_grid = np.fft.ifftn(np.fft.fftn(self.density_grid) * greens_function).real

    def solve_poisson_2(self):
        kx, ky, kz = np.meshgrid(
            np.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            np.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            np.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            indexing='ij'
        )
        sin2_kx = np.sin(kx * self.grid_spacing / 2) ** 2
        sin2_ky = np.sin(ky * self.grid_spacing / 2) ** 2 
        sin2_kz = np.sin(kz * self.grid_spacing / 2) ** 2
        denominator = sin2_kx + sin2_ky + sin2_kz
        denominator[0, 0, 0] = 1

        greens_function = -self.grid_spacing ** 2 / ( 4 * denominator)
        greens_function[0, 0, 0] = 0

        self.potential_grid = np.fft.ifftn(np.fft.fftn(self.density_grid) * greens_function).real

    def compute_forces(self):
        self.force_grid = np.zeros((3, self.grid_size + 1, self.grid_size + 1, self.grid_size + 1))
        
        self.force_grid[0, 1:-1, :, :] = -(self.potential_grid[2:, :, :] - self.potential_grid[:-2, :, :]) / 2
        self.force_grid[1, :, 1:-1, :] = -(self.potential_grid[:, 2:, :] - self.potential_grid[:, :-2, :]) / 2
        self.force_grid[2, :, :, 1:-1] = -(self.potential_grid[:, :, 2:] - self.potential_grid[:, :, :-2]) / 2
        
        self.force_grid[0, 0, :, :] = -(self.potential_grid[1, :, :] - self.potential_grid[-2, :, :]) / 2
        self.force_grid[0, -1, :, :] = self.force_grid[0, 0, :, :]
        self.force_grid[1, :, 0, :] = -(self.potential_grid[:, 1, :] - self.potential_grid[:, -2, :]) / 2
        self.force_grid[1, :, -1, :] = self.force_grid[1, :, 0, :]
        self.force_grid[2, :, :, 0] = -(self.potential_grid[:, :, 1] - self.potential_grid[:, :, -2]) / 2
        self.force_grid[2, :, :, -1] = self.force_grid[2, :, :, 0]

    def update_velocity(self, delta_ln_a, scheme):
        const = 1.5 / (self.a * self.E(self.a)) * (delta_ln_a / 2) * np.exp(self.lna)
        if scheme == 'NGP':
            for n in range(self.num_particles):
                pos = self.positions[n]
                i, j, k = np.floor(pos + 0.5).astype(int)
                i %= self.grid_size
                j %= self.grid_size
                k %= self.grid_size
                for axis in range(3):
                    self.velocities[n, axis] += self.force_grid[axis, i, j, k] * const

        if scheme == 'CIC':
            for n in range(self.num_particles):
                pos = self.positions[n]
                i, j, k = np.floor(pos).astype(int)

                dx, dy, dz = pos - np.array([i, j, k])
                ip1, jp1, kp1 = i + 1, j + 1, k + 1

                w000 = (1 - dx) * (1 - dy) * (1 - dz)
                w100 = dx * (1 - dy) * (1 - dz)
                w010 = (1 - dx) * dy * (1 - dz)
                w001 = (1 - dx) * (1 - dy) * dz
                w110 = dx * dy * (1 - dz)
                w101 = dx * (1 - dy) * dz
                w011 = (1 - dx) * dy * dz
                w111 = dx * dy * dz
                for axis in range(3):
                    self.velocities[n, axis] += (
                        w000 * self.force_grid[axis, i, j, k]+
                        w100 * self.force_grid[axis, ip1, j, k]+
                        w010 * self.force_grid[axis, i, jp1, k]+
                        w001 * self.force_grid[axis, i, j, kp1]+
                        w110 * self.force_grid[axis, ip1, jp1, k]+
                        w101 * self.force_grid[axis, ip1, j, kp1]+
                        w011 * self.force_grid[axis, i, jp1, kp1]+
                        w111 * self.force_grid[axis, ip1, jp1, kp1]
                    ) * const

        if scheme == 'TSC':
            for axis in range(3):
                for pos in self.positions:
                    ix0 = np.floor(pos[0]).astype(int)
                    iy0 = np.floor(pos[1]).astype(int)
                    iz0 = np.floor(pos[2]).astype(int)

                    dx = pos[0] - ix0
                    dy = pos[1] - iy0
                    dz = pos[2] - iz0

                    wx = np.array([0.5 * (1 - dx) ** 2, 0.75 - (dx - 0.5) ** 2, 0.5 * dx ** 2])
                    wy = np.array([0.5 * (1 - dy) ** 2, 0.75 - (dy - 0.5) ** 2, 0.5 * dy ** 2])
                    wz = np.array([0.5 * (1 - dz) ** 2, 0.75 - (dz - 0.5) ** 2, 0.5 * dz ** 2])

                    grid_indices_x = np.clip(np.array([ix0 - 1, ix0, ix0 + 1]), 0, self.grid_size - 1)
                    grid_indices_y = np.clip(np.array([iy0 - 1, iy0, iy0 + 1]), 0, self.grid_size - 1)
                    grid_indices_z = np.clip(np.array([iz0 - 1, iz0, iz0 + 1]), 0, self.grid_size - 1)

                    for xi, wxi in zip(grid_indices_x, wx):
                        for yi, wyi in zip(grid_indices_y, wy):
                            for zi, wzi in zip(grid_indices_z, wz):
                                self.velocities[:, axis] += wxi * wyi * wzi * self.force_grid[axis, xi, yi, zi] * const

    def leapfrog(self, delta_ln_a, scheme):
        self.update_velocity(delta_ln_a, scheme)
        self.positions += self.velocities * delta_ln_a / (self.a**2 * self.E(self.a)) * np.exp(self.lna)
        self.positions %= self.grid_size  # periodic boundary
        self.lna += delta_ln_a
        self.a = np.exp(self.lna)
        self.update_velocity(delta_ln_a, scheme)

    def kinetic(self):
        return 0.5 * np.sum(self.velocities**2)

    def potential(self):
        return 0.5 * np.sum(self.density_grid * self.potential_grid)

    def visualize(self, bins=512):
        # Move positions from GPU to CPU for plotting
        x = self.positions[:, 0]
        y = self.positions[:, 1]

        heatmap, xedges, yedges = np.histogram2d(
            x, y, bins=bins, range=[[0, self.grid_size], [0, self.grid_size]]
        )

        plt.figure(figsize=(8, 8))
        plt.imshow(
            heatmap.T,
            origin='lower',
            extent=[0, self.grid_size, 0, self.grid_size],
            cmap='inferno',
            norm=PowerNorm(gamma=0.4)  # Boost faint filaments
        )

        plt.axis('off')
        plt.tight_layout()
        plt.show()




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
        psi = np.zeros_like(self.potential_grid)
        H = self.hubble()
        dHdp = -1 * (self.hubble(a=self.a) - self.hubble(a=self.a + np.exp(self.lna) * dp)) / dp
        laplacian_coeff = 1 / (self.a * H * self.grid_size) ** 2
        damping_coeff = (1/self.a + dHdp * np.exp(-self.lna) / H + 2 * kappa / (self.a * H))
        for iteration in range(max_iter):
            self.potential_grid += dp * np.exp(self.lna) * psi

            laplacian = np.zeros_like(self.potential_grid)
            phi_xp = np.roll(self.potential_grid, -1, axis=0)
            phi_xm = np.roll(self.potential_grid, 1, axis=0)
            phi_yp = np.roll(self.potential_grid, -1, axis=1)
            phi_ym = np.roll(self.potential_grid, 1, axis=1)
            phi_zp = np.roll(self.potential_grid, -1, axis=2)
            phi_zm = np.roll(self.potential_grid, 1, axis=2)

            laplacian = (phi_xp + phi_xm + phi_yp + phi_ym + phi_zp + phi_zm - 6 * self.potential_grid)
            psi += dp * np.exp(self.lna) * (-damping_coeff * psi + laplacian_coeff * laplacian - laplacian_coeff * self.density_grid)

        return iteration + 1

   
