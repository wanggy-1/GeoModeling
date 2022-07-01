import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy import ndimage
from prettytable import PrettyTable
from PIL import Image, ImageDraw
from pyvistaqt import BackgroundPlotter
from scipy.spatial import distance
import numba
import sys
import pyvista as pv
import time
import multiprocessing
from joblib import Parallel, delayed


class BiharmonicSpline2D:
    """
    2D Bi-harmonic Spline Interpolation.
    """

    def __init__(self, x, y):
        """
        Use coordinates of the known points to calculate the weight "w" in the interpolation function.
        :param x: (numpy.ndarray) - x-coordinates of the known points.
        :param y: (numpy.ndarray) - y-coordinates of the known points.
        """
        self.x = x
        self.y = y
        # Check if the coordinates' shapes are identical.
        if not self.x.shape == self.y.shape:
            raise ValueError("The coordinates' shapes of known points are not identical.")
        # Flatten the coordinates if they are not.
        if self.x.ndim != 1 and self.y.ndim != 1:
            self.x = self.x.ravel(order='C')
            self.y = self.y.ravel(order='C')
        # Calculate the 1D Green function matrix.
        green = np.zeros(shape=[len(self.x), len(self.x)], dtype='float32')
        for i in range(len(x)):
            green[i, :] = np.abs(self.x[i] - self.x) ** 3
        # Calculate weights.
        if np.linalg.matrix_rank(green) == green.shape[0]:  # See if the Green matrix is reversible.
            self.w = np.linalg.inv(green) @ self.y
        else:
            self.w = np.linalg.pinv(green) @ self.y  # Pseudo-inverse.

    def __call__(self, x_new):
        """
        Interpolate new points.
        :param x_new: (numpy.ndarray) - x-coordinates of the new points.
        :return: (numpy.ndarray) - y-coordinates of the new points.
        """
        original_shape = x_new.shape
        # Flatten the coordinates if they are not.
        if x_new.ndim != 1:
            x_new = x_new.ravel(order='C')
        # Calculate the 1D Green function matrix.
        green = np.zeros(shape=[len(x_new), len(self.x)], dtype='float32')
        for i in range(len(x_new)):
            green[i, :] = np.abs(x_new[i] - self.x) ** 3
        # Calculate y-coordinates of new points.
        y_new = green @ self.w
        y_new = y_new.reshape(original_shape, order='C')
        return y_new


class BiharmonicSpline3D:
    """
    3D Bi-harmonic Spline Interpolation.
    """

    def __init__(self, x, y, z):
        """
        Use coordinates of the known points to calculate the weight "w" in the interpolation function.
        :param x: (numpy.ndarray) - x-coordinates of the known points.
        :param y: (numpy.ndarray) - y-coordinates of the known points.
        :param z: (numpy.ndarray) - z-coordinates of the known points.
        """
        self.x = x
        self.y = y
        self.z = z
        # Check if the coordinates' shapes are identical.
        if not self.x.shape == self.y.shape == self.z.shape:
            raise ValueError("The coordinates' shapes of known points are not identical.")
        # Flatten the coordinates if they are not.
        if self.x.ndim != 1 and self.y.ndim != 1 and self.z.ndim != 1:
            self.x = self.x.ravel(order='C')
            self.y = self.y.ravel(order='C')
            self.z = self.z.ravel(order='C')
        # Calculate the 2D Green function matrix.
        delta_x = np.zeros(shape=[len(self.x), len(self.x)], dtype='float32')
        delta_y = np.zeros(shape=[len(self.y), len(self.y)], dtype='float32')
        for i in range(len(x)):
            delta_x[i, :] = self.x[i] - self.x  # Calculate the x-coordinate difference between two points.
            delta_y[i, :] = self.y[i] - self.y  # Calculate the y-coordinate difference between two points.
        mod = np.sqrt(delta_x ** 2 + delta_y ** 2)  # The vector's modulus between two points.
        mod = mod.ravel(order='C')  # Flatten the 2D mod array to 1D.
        green = np.zeros(shape=mod.shape, dtype='float32')  # Initialize the Green function matrix.
        # Calculate the Green function matrix at non-zero points.
        green[mod != 0] = mod[mod != 0] ** 2 * (np.log(mod[mod != 0]) - 1)
        green = green.reshape(delta_x.shape)  # Reshape the matrix to 2-D array shape.
        # Calculate weights.
        if np.linalg.matrix_rank(green) == green.shape[0]:  # See if the Green matrix is reversible.
            self.w = np.linalg.inv(green) @ self.z
        else:
            self.w = np.linalg.pinv(green) @ self.z  # Pseudo-inverse.

    def __call__(self, x_new, y_new):
        """
        Interpolate new points.
        :param x_new: (numpy.ndarray) - x-coordinates of the new points.
        :param y_new: (numpy.ndarray) - y-coordinates of the new points.
        :return: (numpy.ndarray) - z-coordinates of the new points.
        """
        original_shape = x_new.shape
        # Check if the coordinates' shapes are identical.
        if not x_new.shape == y_new.shape:
            raise ValueError("The coordinates' shapes of known points are not identical.")
        # Flatten the coordinates if they are not.
        if x_new.ndim != 1 and y_new.ndim != 1:
            x_new = x_new.ravel(order='C')
            y_new = y_new.ravel(order='C')
        delta_x = np.zeros(shape=[len(x_new), len(self.x)], dtype='float32')
        delta_y = np.zeros(shape=[len(y_new), len(self.y)], dtype='float32')
        for i in range(len(x_new)):
            delta_x[i, :] = x_new[i] - self.x
            delta_y[i, :] = y_new[i] - self.y
        mod = np.sqrt(delta_x ** 2 + delta_y ** 2)  # The vector's modulus between two points.
        mod = mod.ravel(order='C')  # Flatten the 2D mod array to 1D.
        green = np.zeros(shape=mod.shape, dtype='float32')
        green[mod != 0] = mod[mod != 0] ** 2 * (np.log(mod[mod != 0]) - 1)
        green = green.reshape(delta_x.shape)
        # Calculate z-coordinates of new points.
        z_new = green @ self.w
        z_new = z_new.reshape(original_shape, order='C')
        return z_new


class GeoModel:
    def __init__(self, extent, resolution):
        """
        Initialize the 3-D geological model. The current function can only create Regular Grid.
        :param extent: (List of floats) - [X_min, X_max, Y_min, Y_max, Z_min, Z_max]. Extent of the model.
                       Unit: meter.
        :param resolution: (List of floats) - [X_res, Y_res, Z_res]. Model's resolution in each dimension.
                           Unit: meter.
        """
        self.X_min, self.X_max = extent[0], extent[1]
        self.Y_min, self.Y_max = extent[2], extent[3]
        self.Z_min, self.Z_max = extent[4], extent[5]
        self.X_resolution, self.Y_resolution, self.Z_resolution = resolution[0], resolution[1], resolution[2]
        X = np.arange(start=self.X_min, stop=self.X_max + self.X_resolution, step=self.X_resolution, dtype='float32')
        Y = np.arange(start=self.Y_min, stop=self.Y_max + self.Y_resolution, step=self.Y_resolution, dtype='float32')
        Z = np.arange(start=self.Z_min, stop=self.Z_max + self.Z_resolution, step=self.Z_resolution, dtype='float32')
        self.X_points, self.Y_points, self.Z_points = len(X), len(Y), len(Z)
        self.X, self.Y, self.Z = np.meshgrid(X, Y, Z)
        self.rc = np.zeros(shape=self.Z.shape, dtype='float32')  # Reflection coefficient.
        self.vp = np.ones(shape=self.Z.shape, dtype='float32')  # P-wave velocity.
        self.vs = np.ones(shape=self.Z.shape, dtype='float32')  # S-wave velocity.
        self.density = np.ones(shape=self.Z.shape, dtype='float32')  # Density.
        self.Ip = np.zeros(shape=self.Z.shape, dtype='float32')  # P-wave impedance.
        self.Is = np.zeros(shape=self.Z.shape, dtype='float32')  # S-wave impedance.
        self.seis = np.array([])  # Synthetic seismic data.
        self.fm = np.zeros(shape=self.Z.shape, dtype='float32')  # The fault marker.
        self.channel_facies = np.zeros(shape=self.Z.shape, dtype='float32')  # The channel sub-facies marker.
        self.lith_facies = np.zeros(shape=self.Z.shape, dtype='float32')  # The lithology facies marker.
        self.channel = []  # List of channels.
        self.oxbow = []  # List of oxbows.
        self.autocrop = {}  # Dictionary of z coordinates used for auto-cropping the model.
        # Create a table of lithology parameters.
        index = ['basalt', 'diabase', 'fine silty mudstone', 'oil shale', 'mudstone', 'silty mudstone', 'lime dolomite',
                 'pelitic dolomite', 'siltstone', 'pebbled sandstone', 'conglomerate', 'fine sandstone',
                 'pelitic siltstone']  # Lithology names.
        column = ['vp_miu', 'vp_sigma', 'vs_miu', 'vs_sigma', 'rho_miu', 'rho_sigma']  # Parameter names.
        data = np.array([[3.38, 0.20, 1.96, 0.15, 2.15, 0.18],
                         [5.14, 0.21, 3.01, 0.13, 2.67, 0.20],
                         [3.72, 0.19, 2.33, 0.18, 2.47, 0.21],
                         [3.85, 0.18, 2.47, 0.16, 2.42, 0.19],
                         [4.78, 0.23, 3.01, 0.20, 2.70, 0.17],
                         [5.43, 0.15, 3.47, 0.18, 2.60, 0.20],
                         [5.70, 0.18, 3.09, 0.16, 2.65, 0.18],
                         [5.61, 0.16, 3.13, 0.15, 2.57, 0.22],
                         [4.24, 0.20, 2.69, 0.18, 2.28, 0.17],
                         [4.35, 0.17, 2.73, 0.20, 2.22, 0.18],
                         [5.54, 0.18, 3.01, 0.21, 2.56, 0.21],
                         [5.26, 0.20, 3.26, 0.18, 2.59, 0.21],
                         [4.58, 0.18, 2.71, 0.16, 2.34, 0.19]], dtype='float32')  # Parameter values.
        self.lith_param = pd.DataFrame(data=data, index=index, columns=column)  # Lithology parameter data-frame.
        # Print model's information.
        print('Model extent:')
        print('X range: %.2fm-%.2fm' % (self.X_min, self.X_max))
        print('Y range: %.2fm-%.2fm' % (self.Y_min, self.Y_max))
        print('Z range: %.2fm-%.2fm' % (self.Z_min, self.Z_max))
        print('Model resolution (XYZ): [%.2fm x %.2fm x %.2fm]' %
              (self.X_resolution, self.Y_resolution, self.Z_resolution))
        print('Model points (XYZ): [%d x %d x %d]' % (self.X_points, self.Y_points, self.Z_points))

    def add_rc(self, rc_range=None):
        """
        Add random reflection coefficient (rc) to the model.
        :param rc_range: (List of floats) - [min, max]. Default is [-1, 1].
                         The range of reflection coefficient.
        """
        if rc_range is None:
            rc_min, rc_max = -1, 1
        else:
            rc_min, rc_max = rc_range[0], rc_range[1]
        rdm = (rc_max - rc_min) * np.random.random_sample((self.Z_points,)) + rc_min
        rc = np.ones(shape=self.Z.shape, dtype='float32')
        for i in range(self.Z_points):
            rc[:, :, i] = rc[:, :, i] * rdm[i]
        self.rc = rc
        # Set the first and the last rc to 0.
        self.rc[:, :, 0] *= 0
        self.rc[:, :, -1] *= 0

    def add_rho_v(self, h_layer_range=None, vp_range=None, vs_range=None, rho_range=None):
        """
        Add density, p-wave velocity and s-wave velocity.
        :param h_layer_range: (List of floats) - Default is [10, 20% * Z_range]. The range of layer thickness.
        :param vp_range: (List of floats) - Default is [2500, 5500]. The range of p-wave velocity (m/s).
        :param vs_range: (List of floats) - Default is [1440, 3175]. The range of s-wave velocity (m/s).
        :param rho_range: (List of floats) - Default is [2.2, 2.7]. The range of density (g/cm3).
        """
        # Initialization.
        depth_top = self.Z_min  # The initial layer's top depth.
        ind_bottom = 0  # The initial array index of layer's bottom depth.
        if vp_range is None:
            vp_range = [2.5, 5.5]
        if vs_range is None:
            vs_range = [1.44, 3.175]
        if rho_range is None:
            rho_range = [2.2, 2.7]
        # Assign velocity and density from the top to the bottom.
        while ind_bottom < self.Z_points - 1:
            # Set layer thickness randomly.
            if h_layer_range is None:
                h_layer = random.uniform(10, 0.2 * (self.Z_max - self.Z_min))
            else:
                h_layer = random.uniform(h_layer_range[0], h_layer_range[1])
            # Compute the layer's bottom depth.
            depth_bottom = depth_top + h_layer
            # Layer's bottom depth can not be greater than Z_max.
            if depth_bottom > self.Z_max:
                depth_bottom = self.Z_max
            # Compute array index.
            ind_top = int((depth_top - self.Z_min) // self.Z_resolution)  # Layer's top depth index.
            ind_bottom = int((depth_bottom - self.Z_min) // self.Z_resolution)  # Layer's bottom depth index.
            # Assign velocity and density.
            vp_layer = random.uniform(vp_range[0], vp_range[1])
            vs_layer = random.uniform(vs_range[0], vs_range[1])
            rho_layer = random.uniform(rho_range[0], rho_range[1])
            if vp_layer / vs_layer < 1.5:
                vs_layer = vp_layer / 1.5
            if vp_layer / vs_layer > 2.3:
                vs_layer = vp_layer / 2.3
            self.vp[:, :, ind_top:ind_bottom + 1] *= vp_layer  # P-wave velocity.
            self.vs[:, :, ind_top:ind_bottom + 1] *= vs_layer  # S-wave velocity.
            self.density[:, :, ind_top:ind_bottom + 1] *= rho_layer  # Density.
            # Update layer top depth.
            depth_top = (ind_bottom + 1) * self.Z_resolution

    def compute_impedance(self):
        """
        Compute P-wave impedance and S-wave impedance.
        """
        self.Ip = self.vp * self.density
        self.Is = self.vs * self.density

    def compute_rc(self):
        """
        Compute reflection coefficient from P-wave impedance.
        RC = (Ii+1 - Ii) / (Ii+1 + Ii)
        """
        for i in range(self.Ip.shape[2] - 1):
            sys.stdout.write('\rComputing reflection coefficient: %.2f%%' % ((i + 1) / (self.Ip.shape[2] - 1) * 100))
            self.rc[:, :, i] = (self.Ip[:, :, i + 1] - self.Ip[:, :, i]) / (self.Ip[:, :, i + 1] + self.Ip[:, :, i])
        sys.stdout.write('\n')

    def make_synseis(self, A=100, fm=30, dt=0.002, wavelet_len=0.1, plot_wavelet=True):
        """
        Make synthetic seismic data.
        :param A: (float) - Default is 100. The maximum amplitude of ricker wavelet.
        :param fm: (Float) - Default is 30Hz. Dominant frequency of the wavelet.
        :param dt: (Float) - Default is 0.002s. Sampling time interval of the wavelet.
        :param wavelet_len: (Float) - Default is 0.1s. Time duration of the wavelet.
        :param plot_wavelet: (Bool) - Default is True. Whether to visualize the wavelet.
        """
        self.seis = np.zeros(shape=self.rc.shape, dtype='float32')
        t = np.arange(-wavelet_len / 2, wavelet_len / 2, dt, dtype='float32')
        ricker = A * (1 - 2 * math.pi ** 2 * fm ** 2 * t ** 2) * np.exp(-math.pi ** 2 * fm ** 2 * t ** 2)
        for i in range(self.rc.shape[0]):
            for j in range(self.rc.shape[1]):
                sys.stdout.write('\rGenerating synthetic seismic data: %.2f%%' %
                                 ((i*self.rc.shape[1]+j+1) / (self.rc.shape[0]*self.rc.shape[1]) * 100))
                self.seis[i, j, :] = np.convolve(ricker, self.rc[i, j, :], mode='same')
        sys.stdout.write('\n')
        if plot_wavelet:
            plt.figure()
            plt.style.use('bmh')
            plt.plot(t, ricker, lw=2)
            plt.xlabel('t(s)')
            plt.ylabel('y')
            plt.show()

    def add_noise(self, noise_type='uniform', ratio=0.05):
        """
        Add random noise to synthetic seismic data.
        :param noise_type: (String) - The noise type. Default is uniform random noise.
                           Options are:
                           1. 'uniform': generate random noise with uniform distribution.
                           2. 'gaussian': generate random noise with normal distribution.
        :param ratio: (Float) - Default is 5%. The noise amplitude to seismic absolute maximum amplitude ratio.
        """
        print(f'Adding {noise_type} noise...')
        seis_min, seis_max = np.amin(np.abs(self.seis)), np.amax(np.abs(self.seis))
        seis_mean = np.average(np.abs(self.seis))  # Get the average amplitude of synthetic seismic data.
        print('Seismic data [min, max, mean]: [%.2f, %.2f, %.2f]' % (seis_min, seis_max, seis_mean))
        if noise_type == 'uniform':  # Add uniform random noise.
            # Generate noise.
            noise_min, noise_max = [-ratio * seis_mean, ratio * seis_mean]  # The range of uniform random noise.
            print('Noise range: [%.2f, %.2f]' % (noise_min, noise_max))
            noise = (noise_max - noise_min) * np.random.random_sample(self.seis.shape) + noise_min
        if noise_type == 'gaussian':  # Add random noise with normal distribution.
            # Generate noise.
            noise_std = ratio * seis_mean  # The standard deviation of random noise.
            print('Noise info [mean, std]: [0.00, %.2fs]' % noise_std)
            noise = np.random.normal(loc=0, scale=noise_std, size=self.seis.shape)
        # Add noise to seismic data.
        noise = noise.astype('float32')  # Change data type.
        self.seis = self.seis + noise

    def add_fold(self, N=10, miu_X_range=None, miu_Y_range=None, sigma_range=None, A_range=None, sync=False):
        """
        Simulate folds with a combination of Gaussian functions.
        :param N: (Integer) - Default is 10. Control the number of Gaussian functions.
        :param miu_X_range: (List of floats) - [min, max]. Default is [X_min, X_max).
                            Center coordinate's (X-coordinate) range of the Gaussian function.
        :param miu_Y_range: (List of floats) - [min, max]. Default is [Y_min, Y_max).
                            Center coordinate's (Y-coordinate) range of the Gaussian function.
        :param sigma_range: (List of floats) - [min, max].
                            Default is [10% * min(X_range, Y_range), 20% * max(X_range, Y_range)).
                            The half-width's range of the Gaussian function.
        :param A_range: (List of floats) - [min, max].
                        Default is [5% * Z_range, 10% * Z_range).
                        The amplitude's range of the Gaussian function.
        :param sync: (Bool) - Default is False: deeper layers have more uplift.
                     If True: layers with same XY have same uplift.
        """
        t_start = time.perf_counter()
        sys.stdout.write('Simulating folding structure...')
        Gaussian_sum = 0  # Initialize the summation of Gaussian function.
        fold_parameter = PrettyTable()  # For visualizing parameters
        fold_parameter.field_names = ['Fold Number', 'miu_X', 'miu_Y', 'sigma', 'Amplitude']
        fold_parameter.float_format = '.2'
        for i in range(N):
            if miu_X_range is None:
                miu_X = random.uniform(self.X_min, self.X_max)
            else:
                miu_X = random.uniform(miu_X_range[0], miu_X_range[1])
            if miu_Y_range is None:
                miu_Y = random.uniform(self.Y_min, self.Y_max)
            else:
                miu_Y = random.uniform(miu_Y_range[0], miu_Y_range[1])
            if sigma_range is None:
                sigma = random.uniform(0.1 * min(self.X_max - self.X_min, self.Y_max - self.Y_min),
                                       0.2 * min(self.X_max - self.X_min, self.Y_max - self.Y_min))
            else:
                sigma = random.uniform(sigma_range[0], sigma_range[1])
            if A_range is None:
                A = random.uniform(0.05 * (self.Z_max - self.Z_min), 0.1 * (self.Z_max - self.Z_min))
            else:
                A = random.uniform(A_range[0], A_range[1])
            # The Gaussian function.
            f_Gaussian = A * np.exp(-1 * ((self.X - miu_X) ** 2 + (self.Y - miu_Y) ** 2) / (2 * sigma ** 2))
            Gaussian_sum += f_Gaussian  # Combine the Gaussian functions.
            fold_parameter.add_row([i + 1, miu_X, miu_Y, sigma, A])  # Visualizing parameters.
        # Shift the Z-coordinates vertically.
        if sync is False:
            self.Z = self.Z - self.Z / self.Z.max() * Gaussian_sum
        else:
            self.Z = self.Z - Gaussian_sum
        # Limit the model in defined extent range.
        self.Z[self.Z > self.Z_max] = self.Z_max
        self.Z[self.Z < self.Z_min] = self.Z_min
        t_end = time.perf_counter()
        sys.stdout.write('Done.\n')
        print('Simulation time: %.2fs' % (t_end - t_start))
        print(fold_parameter)

    def add_fault(self, N=3, reference_point_range=None, phi_range=None, theta_range=None,
                  d_max_range=None, lx_range=None, ly_range=None, gamma_range=None, beta_range=None,
                  curved_fault_surface=True, n_perturb=20, perturb_range=None,
                  mark_faults=False,
                  computation_mode='parallel'):
        """
        Simulate faults.
        :param N: (Integer) - Default is 3. Number of faults in a model.
        :param reference_point_range: (List of floats) - [X0_min, X0_max, Y0_min, Y0_max, Z0_min, Z0_max].
                                      Default is X0 = [X_min, X_max),
                                                 Y0 = [Y_min, Y_max),
                                                 Z0 = [Z_min, Z_max).
                                      The 3-D coordinates' range of a fault surface's center point.
        :param phi_range: (List of floats) - [min, max]. Default is [0, 360). The range of strike angle.
        :param theta_range: (List of floats) - [min, max]. Default is [0, 90). The range of dip angle.
        :param d_max_range: (List of floats) - [min, max]. Default is [10% * y_range, 30% * y_range).
                            The range of maximum displacement on the fault surface.
                            The customized range should be fractions of y_range (e.g. [0.1, 1.1]), where y_range is the
                            model's length in y-direction (dip direction) of fault surface coordinates.
        :param lx_range: (List of floats) - [min, max]. Default is [50% * x_range, 100% * x_range).
                         The range of strike direction axis' length of the elliptic displacement field on fault surface.
                         The customized range should be fractions of x_range (e.g. [0.1, 1.1]), where x_range is the
                         model's length in x-direction (strike direction) of fault surface coordinates.
        :param ly_range: (List of floats) - [min, max]. Default is [50% * y_range, 100% * y_range).
                         The range of dip direction axis' length of the elliptic displacement field on fault surface.
                         The customized range should be fractions of y_range (e.g. [0.1, 1.1]), where y_range is the
                         model's length in y-direction (dip direction) of fault surface coordinates.
        :param gamma_range: (List of floats) - [min, max]. Default is [10% * z_range, 50% * z_range).
                            The range of reverse drag radius.
                            The customized range should be fractions of z_range (e.g. [0.1, 1.1]), where z_range is the
                            model's length in z-direction (normal direction) of the fault surface coordinates.
        :param beta_range: (List of floats) - [min, max]. Default is [0.5, 1).
                           The range of hanging-wall's displacement / d_max.
        :param curved_fault_surface: (Bool) - Default is True.
                                     Whether to create curved fault surface.
        :param n_perturb: (Integer) - Default is 20. Number of perturbation points near the fault surface.
        :param perturb_range: (List of floats) - Default is [-5% * z_range, 5% * z_range).
                              The range of perturbation points' z coordinates.
                              The customized range should be fractions of z_range (e.g. [-0.05, 0.05]), where z_range is
                              the model's length in z-direction (normal direction) of the fault surface coordinates.
        :param mark_faults: (Bool) - Default is False. If True, mark faults with label "1" and others with label "0".
        :param computation_mode: (String) - Default is "parallel", which is to break down the model's coordinate arrays
                                 into slices and simulate curved faults in parallel.
                                 'non-parallel' takes the whole coordinate arrays as input to simulate curved faults.
                                 Notice that when the model size is small (e.g. 32 x 32 x 32), taking the whole
                                 coordinate arrays as input will be faster.
                                 In addition, when the memory space is not enough, use the 'parallel' mode may solve the
                                 problem.
        """
        t_start = time.perf_counter()
        if curved_fault_surface:
            if computation_mode != 'parallel' and computation_mode != 'non-parallel':
                raise ValueError("'computation_mode' must be 'parallel' or 'non-parallel'.")
            else:
                sys.stdout.write(f'Simulating curved fault in {computation_mode} mode...')
        else:
            sys.stdout.write('Simulating planar fault...')
        fault_parameter = PrettyTable()  # For visualizing parameters.
        fault_parameter.field_names = ['Fault Number', 'X0', 'Y0', 'Z0', 'phi', 'theta', 'dmax', 'lx', 'ly',
                                       'gamma', 'beta']
        fault_parameter.float_format = '.2'
        for n in range(N):
            if reference_point_range is None:
                X0, Y0, Z0 = random.uniform(self.X.min(), self.X.max()), \
                             random.uniform(self.Y.min(), self.Y.max()), \
                             random.uniform(self.Z.min(), self.Z.max())
            else:
                X0, Y0, Z0 = random.uniform(reference_point_range[0], reference_point_range[1]), \
                             random.uniform(reference_point_range[2], reference_point_range[3]), \
                             random.uniform(reference_point_range[4], reference_point_range[5])
            if phi_range is None:
                phi = random.uniform(0, 360)
            else:
                phi = random.uniform(phi_range[0], phi_range[1])
            if theta_range is None:
                theta = random.uniform(0, 90)
            else:
                theta = random.uniform(theta_range[0], theta_range[1])
            phi, theta = [math.radians(phi), math.radians(theta)]  # Convert from angle to radian.
            R = [[math.sin(phi), - math.cos(phi), 0],  # Rotation matrix.
                 [math.cos(phi) * math.cos(theta), math.sin(phi) * math.cos(theta), math.sin(theta)],
                 [math.cos(phi) * math.sin(theta), math.sin(phi) * math.sin(theta), -math.cos(theta)]]
            R = np.array(R, dtype='float32')
            # The points' global coordinates relative to the reference point.
            cor_g = np.array([(self.X - X0).ravel(order='C'),
                              (self.Y - Y0).ravel(order='C'),
                              (self.Z - Z0).ravel(order='C')], dtype='float32')
            # Coordinates rotation.
            # "x" is the strike direction, "y" is the dip direction and "z" is the normal direction.
            [x, y, z] = R @ cor_g
            x = x.reshape(self.X.shape, order='C')
            y = y.reshape(self.Y.shape, order='C')
            z = z.reshape(self.Z.shape, order='C')
            if lx_range is None:
                lx = random.uniform(0.5 * (x.max() - x.min()), 1.0 * (x.max() - x.min()))
            else:
                lx = random.uniform(lx_range[0] * (x.max() - x.min()), lx_range[1] * (x.max() - x.min()))
            if ly_range is None:
                ly = random.uniform(0.1 * (y.max() - y.min()), 0.5 * (y.max() - y.min()))
            else:
                ly = random.uniform(ly_range[0] * (y.max() - y.min()), ly_range[1] * (y.max() - y.min()))
            r = np.sqrt((x / lx) ** 2 + (y / ly) ** 2)  # The elliptic surface along the fault plane.
            r[r > 1] = 1  # To make the displacement = 0 outside the elliptic surface's boundary.
            # The elliptic displacement field along the fault plane.
            if d_max_range is None:
                d_max = random.uniform(0.1 * (y.max() - y.min()), 0.3 * (y.max() - y.min()))
            else:
                d_max = random.uniform(d_max_range[0] * (y.max() - y.min()), d_max_range[1] * (y.max() - y.min()))
            d = 2 * d_max * (1 - r) * np.sqrt((1 + r) ** 2 / 4 - r ** 2)
            f = 0  # Define fault surface (0 for plane surface).
            # Create curved fault surface.
            if curved_fault_surface:
                # Randomly choose the 3-D coordinates of perturbation points.
                if perturb_range is None:
                    perturb_range = [-0.05 * (z.max() - z.min()), 0.05 * (z.max() - z.min())]
                else:
                    perturb_range = [perturb_range[0] * (z.max() - z.min()), perturb_range[1] * (z.max() - z.min())]
                x_perturb = (x.max() - x.min()) * np.random.random_sample((n_perturb,)) + x.min()
                y_perturb = (y.max() - y.min()) * np.random.random_sample((n_perturb,)) + y.min()
                z_perturb = \
                    (perturb_range[1] - perturb_range[0]) * np.random.random_sample((n_perturb,)) + perturb_range[0]
                # Use the perturbation points to calculate the parameters of Bi-harmonic Spline interpolator.
                interpolator = BiharmonicSpline3D(x_perturb, y_perturb, z_perturb)
                # Interpolate a curved fault surfaces.
                if computation_mode == 'parallel':
                    n_cores = multiprocessing.cpu_count()  # Get the number of cpu cores.
                    f = Parallel(n_jobs=n_cores)(delayed(compute_f_parallel)(i, x, y, interpolator)
                                                 for i in range(x.shape[0]))  # Compute in parallel.
                    f = np.array(f, dtype='float32')
                else:
                    f = interpolator(x, y)
            # Mark faults.
            if mark_faults:
                z_resolution = (z.max() - z.min()) / (self.Z_points - 1)
                ind = (np.abs(z - f) < z_resolution) & (d > 0)
                self.fm[ind] = 1
            # Nonlinear scalar function that decreases along z-axis from fault surface.
            if gamma_range is None:
                gamma = random.uniform(0.1 * (z.max() - z.min()), 0.5 * (z.max() - z.min()))
            else:
                gamma = random.uniform(gamma_range[0] * (z.max() - z.min()), gamma_range[1] * (z.max() - z.min()))
            alpha = (1 - np.abs(z - f) / gamma) ** 2
            # Initialize the displacement array.
            Dx = 0  # Strike displacement.
            Dy = np.zeros(shape=y.shape, dtype='float32')  # Dip displacement
            Dz = 0  # Normal displacement.
            # Calculate volumetric displacement of the hanging-wall.
            if beta_range is None:
                beta = random.uniform(0.5, 1)
            else:
                beta = random.uniform(beta_range[0], beta_range[1])
            Dy[(z > f) & (z <= f + gamma)] = beta * d[(z > f) & (z <= f + gamma)] * alpha[(z > f) & (z <= f + gamma)]
            # Calculate volumetric displacement of the foot-wall.
            Dy[(z >= f - gamma) & (z <= f)] = \
                (beta - 1) * d[(z >= f - gamma) & (z <= f)] * alpha[(z >= f - gamma) & (z <= f)]
            # Add fault displacement.
            x = x + Dx
            y = y + Dy
            if curved_fault_surface:
                if computation_mode == 'parallel':
                    Dz = Parallel(n_jobs=n_cores)(delayed(compute_Dz_parallel)(i, x, y, f, interpolator)
                                                  for i in range(x.shape[0]))  # Compute in parallel.
                    Dz = np.array(Dz, dtype='float32')
                else:
                    Dz = interpolator(x, y) - f
            z = z + Dz
            # Transform back to global coordinate.
            cor_f = np.array([x.ravel(order='C'),  # "cor_f" is the points' fault-plane coordinates.
                              y.ravel(order='C'),
                              z.ravel(order='C')], dtype='float32')
            [X_faulted, Y_faulted, Z_faulted] = np.linalg.inv(R) @ cor_f + np.array([[X0], [Y0], [Z0]], dtype='float32')
            self.X = X_faulted.reshape(self.X.shape, order='C')
            self.Y = Y_faulted.reshape(self.Y.shape, order='C')
            self.Z = Z_faulted.reshape(self.Z.shape, order='C')
            fault_parameter.add_row([n + 1, X0, Y0, Z0, phi * 180 / math.pi, theta * 180 / math.pi,
                                     d_max, lx, ly, gamma, beta])
        # Limit model in defined extent range.
        self.X[self.X > self.X_max] = self.X_max
        self.X[self.X < self.X_min] = self.X_min
        self.Y[self.Y > self.Y_max] = self.Y_max
        self.Y[self.Y < self.Y_min] = self.Y_min
        self.Z[self.Z > self.Z_max] = self.Z_max
        self.Z[self.Z < self.Z_min] = self.Z_min
        t_end = time.perf_counter()
        sys.stdout.write('Done.\n')
        print('Simulation time: %.2fs' % (t_end - t_start))
        print(fault_parameter)

    def add_meander(self, N, X_pos_range=None, Y_pos_range=None, Z_pos_range=None, strike_range=None,
                    delta_s=None, n_bends_range=None, perturb_range=None, nit_range=None, dt=None, save_it=10,
                    W_range=None, D_range=None, kl_range=None, Cf_range=None, pad_up=None, pad_down=None,
                    h_lag_range=None, h_levee_range=None, w_levee_range=None, h_oxbow_mud_range=None,
                    critical_distance=None, show_migration=False, figure_title=None, mode='simple'):
        """
        Simulate meandering river migration and deposition.
        :param N: (Integer) - Number of meandering rivers.
        :param X_pos_range: (List of floats) - [min, max].
                            Default is [0, s_init(straight center-line's length) - X_range).
                            Range of river center-line's x-coordinate which the 3D model starts at.
        :param Y_pos_range: (List of floats) - [min, max]. Default is [Y_min + 10%*Y_range, Y_max - 10%*Y_range).
                            Range of the initial straight center-line's y-coordinate.
        :param Z_pos_range: (List of floats) - [min, max]. Default is [Z_min + 10%*Z_range, Z_max - 10%*Z_range).
                            Range of the initial straight center-line's z-coordinate.
        :param strike_range: (List of floats) - [min, max]. Default is [0, 360).
                              Range of the river's strike angle (Relative to x-direction).
        :param delta_s: (Floats) - Default is self-adaptive according to the length of initial straight center-line
                        (s_init // 600).
                        Sampling interval along the river's center-line.
        :param n_bends_range: (List of integers) - [min, max]. Default is [10, 20).
                              Range of the number of bends in the initial center-line.
        :param perturb_range: (List of floats) - [min, max]. Default is [200, 500). Range of perturbation amplitude for
                              center-line initialization.
        :param nit_range: (List of integers) - [min, max]. Default is [500, 2000). Range of the number of iteration.
        :param dt: (Float) - Default is 0.1. Time interval of the migration (year).
        :param save_it: (Integer) - Default is 10. Save center-line for every "save_it" iteration.
        :param W_range: (List of floats) - [min, max]. Default is [50, 1500).
                        Range of the river's width (assuming uniform width).
        :param D_range: (List of floats) - [min, max]. Default is [20, 200). Range of the river's depth.
        :param kl_range: (List of floats) - [min, max]. Default is [10, 50).
                         Range of the migration rate constant (m/year).
        :param Cf_range: (List of floats) - [min, max]. Default is [0.05, 0.1). Range of the friction factor.
        :param pad_up: (Integer) - Default is 5. Number of padding points at upstream to fix the center-line.
        :param pad_down: (Integer) - Default is 0. Number of padding points at downstream to fix the center-line.
        :param h_lag_range: (List of floats) - [min, max]. Default is [20% * Depth, 50% * Depth).
                            The maximum thickness of riverbed's lag deposits.
        :param h_levee_range: (List of floats) - [min, max]. Default is [0.1, 1).
                              Range of the maximum levee thickness per event.
        :param w_levee_range: (List of floats) - [min, max]. Default is [2 * channel width, 6 * channel width).
                              Range of the levee width.
        :param h_oxbow_mud_range: (List of floats) - [min, max]. Default is [50% * (D - h_lag), 80% * (D - h_lag)].
                                  The range of oxbow lake mudstone thickness. It is a fraction of (D - h_lag).
                                  For example, enter [0.3, 0.6] will randomly select the mudstone thickness from
                                  30% * (D - h_lag) to 60% * (D - h_lag).
        :param critical_distance: (Float) - Default is river width.The critical distance. Cutoff occurs when distance
                                  of two points on center-line is shorter than (or equal to) the critical distance.
        :param show_migration: (Bool) - Default is False. If True, show river migration progress on X-Y plane.
        :param figure_title: (String) - Default is "Meandering River Migration". The title of river migration figure.
        :param mode: (String) - Default is "simple". Simulation mode ["simple", "complex"].
                     For "simple" mode, just simulate the riverbed lag deposit of the most recent meandering river after
                     the last migration.
                     For "complex" mode, simulate the following deposition sub-facies:
                     1. riverbed sub-facies including lag deposit and point-bar deposit.
                     2. natural levee sub-facies.
                     3. oxbow-lake sub-facies.
        """
        if mode != 'simple' and mode != 'complex':
            raise ValueError(f"No such mode as '{mode}', choose 'simple' mode or 'complex' mode.")
        t_start = time.perf_counter()
        print(f'Simulating meandering channel in {mode} mode...')
        # Make table to display river parameters.
        meander_parameter = PrettyTable()
        if mode == 'simple':
            meander_parameter.field_names = ['Meander number', 'X', 'Y', 'Z', 'strike', 'width', 'depth', 's_init',
                                             'n_bends', 'iteration', 'dt (year)', 'migration rate constant (m/year)',
                                             'friction factor', 'h_lag (m)']
        if mode == 'complex':
            meander_parameter.field_names = ['Meander number', 'X', 'Y', 'Z', 'strike', 'width', 'depth', 's_init',
                                             'n_bends', 'iteration', 'dt (year)', 'migration rate constant (m/year)',
                                             'friction factor', 'h_lag (m)', 'h_levee (m)', 'w_levee (m)']
        meander_parameter.float_format = '.2'
        # Simulation begins.
        for n in range(N):  # Number of rivers.
            # Print progress.
            print('Channel[%d/%d]:' % (n + 1, N))
            # Initialize markers.
            marker = np.zeros(self.Z.shape, dtype='float32')
            # Assign parameters.
            if strike_range is None:  # River strike.
                strike = random.uniform(0, 360)
            else:
                strike = random.uniform(strike_range[0], strike_range[1])
            if n_bends_range is None:  # Initial number of bends.
                n_bends = random.randint(10, 20)
            else:
                n_bends = random.randint(n_bends_range[0], n_bends_range[1])
            if perturb_range is None:  # The perturbation amplitude for river center-line initialization.
                perturb_range = [200, 500]
            if nit_range is None:  # Number of iteration.
                nit = random.randint(500, 2000)
            else:
                nit = random.randint(nit_range[0], nit_range[1])
            if dt is None:  # Migration time interval.
                dt = 0.1
            if W_range is None:  # River width.
                W = random.uniform(50, 1500)
            else:
                W = random.uniform(W_range[0], W_range[1])
            if D_range is None:  # River maximum depth.
                D = random.uniform(20, 200)
            else:
                D = random.uniform(D_range[0], D_range[1])
            if D > W:  # Avoid that river depth is greater than river width.
                D = W
            if h_lag_range is None:  # Riverbed lag deposit thickness.
                h_lag = random.uniform(0.2 * D, 0.5 * D)
            else:
                h_lag = random.uniform(h_lag_range[0], h_lag_range[1])
            if h_lag > D:
                h_lag = D  # Lag deposit thickness can not be greater than channel's maximum depth.
            if h_levee_range is None:  # Natural levee thickness per event.
                h_levee = random.uniform(0.1, 1)
            else:
                h_levee = random.uniform(h_levee_range[0], h_levee_range[1])
            if w_levee_range is None:  # Natural levee width.
                w_levee = random.uniform(2 * W, 6 * W)
            else:
                w_levee = random.uniform(w_levee_range[0], w_levee_range[1])
            if h_oxbow_mud_range is None:
                h_oxbow_mud = random.uniform((D - h_lag) * 0.5, (D - h_lag) * 0.8)
            else:
                h_oxbow_mud = random.uniform(h_oxbow_mud_range[0] * (D - h_lag), h_oxbow_mud_range[1] * (D - h_lag))
            if critical_distance is None:  # Cutoff critical distance.
                critical_distance = W
            if kl_range is None:  # Migration rate constant.
                kl = random.uniform(10, 50)
            else:
                kl = random.uniform(kl_range[0], kl_range[1])
            if Cf_range is None:  # Friction factor.
                Cf = random.uniform(0.05, 0.1)
            else:
                Cf = random.uniform(Cf_range[0], Cf_range[1])
            if pad_up is None:  # Number of upstream padding points.
                pad_up = 5
            if pad_down is None:  # Number of downstream padding points.
                pad_down = 0
            s_init = 5.0 * n_bends * W  # The length of the initial straight center-line.
            if delta_s is None:  # River center-line sampling interval.
                delta_s = s_init // 600  # The default self-adaptive sampling interval (about 600 segments).
            # Make sure that the center-line still crosses the model's X-Y plane after any kind of rotation.
            patch = math.sqrt((self.X_max - self.X_min) ** 2 + 4 * (self.Y_max - self.Y_min) ** 2) - (self.X_max -
                                                                                                      self.X_min)
            if s_init < self.X_max - self.X_min + patch:
                s_init = self.X_max - self.X_min + patch
            if X_pos_range is None:
                X_pos = random.uniform(patch / 2, s_init + self.X_min - self.X_max - patch / 2)
            else:
                X_pos = random.uniform(X_pos_range[0], X_pos_range[1])
                if X_pos < patch / 2:
                    X_pos = patch / 2
                if X_pos > s_init + self.X_min - self.X_max - patch / 2:
                    X_pos = s_init + self.X_min - self.X_max - patch / 2
            if Y_pos_range is None:
                Y_pos = random.uniform(self.Y_min + 0.1 * (self.Y_max - self.Y_min),
                                       self.Y_max - 0.1 * (self.Y_max - self.Y_min))
            else:
                Y_pos = random.uniform(Y_pos_range[0], Y_pos_range[1])
            if Z_pos_range is None:
                Z_pos = random.uniform(self.Z_min + 0.1 * (self.Z_max - self.Z_min),
                                       self.Z_max - 0.1 * (self.Z_max - self.Z_min))
            else:
                Z_pos = random.uniform(Z_pos_range[0], Z_pos_range[1])
            # Add parameters to table.
            if mode == 'simple':
                meander_parameter.add_row([n + 1, X_pos, Y_pos, Z_pos, strike, W, D, s_init, n_bends, nit, dt, kl, Cf,
                                           h_lag])
            if mode == 'complex':
                meander_parameter.add_row([n + 1, X_pos, Y_pos, Z_pos, strike, W, D, s_init, n_bends, nit, dt, kl, Cf,
                                           h_lag, h_levee, w_levee])
            # Print river parameter table.
            print(meander_parameter)
            # Initialize river list.
            centerline = []
            # Initialize oxbow-lake list.
            oxbow_per_channel = []
            X_ctl, Y_ctl, Z_ctl = initialize_centerline(s_init, Y_pos, Z_pos, delta_s, n_bends, perturb_range)
            # Re-sample center-line so that delta_s is roughly constant.
            X_ctl, Y_ctl, Z_ctl = resample_centerline(X_ctl, Y_ctl, Z_ctl, delta_s)
            # Save current river parameters.
            centerline.append(Channel(X_ctl, Y_ctl, Z_ctl, W, D))
            # Start river migration.
            for it in range(nit):
                # Compute derivative of x (dx), y (dy), center-line's length (s) and distances between two
                # consecutive points along center-line.
                dx, dy, s, ds = compute_curvelength(X_ctl, Y_ctl)
                # Compute curvatures at each points on center-line.
                c = compute_curvature(X_ctl, Y_ctl)
                # Compute sinuosity.
                sinuosity = s / (X_ctl[-1] - X_ctl[0])
                # Compute migration rate of each point.
                R1 = compute_migration_rate(curv=c, ds=ds, W=W, kl=kl, Cf=Cf, D=D, pad_up=pad_up, pad_down=pad_down)
                # Adjust migration rate.
                R1 = sinuosity ** (-2 / 3.0) * R1
                # Compute coordinates after migration.
                ns = len(R1)
                dx_ds = dx[pad_up:ns - pad_down] / ds[pad_up:ns - pad_down]
                dy_ds = dy[pad_up:ns - pad_down] / ds[pad_up:ns - pad_down]
                X_ctl[pad_up:ns - pad_down] = X_ctl[pad_up:ns - pad_down] + R1[pad_up:ns - pad_down] * dy_ds * dt
                Y_ctl[pad_up:ns - pad_down] = Y_ctl[pad_up:ns - pad_down] - R1[pad_up:ns - pad_down] * dx_ds * dt
                # Re-sample center-line so that delta_s is roughly constant.
                X_ctl, Y_ctl, Z_ctl = resample_centerline(X_ctl, Y_ctl, Z_ctl, delta_s)
                # Find and execute cutoff.
                X_ox, Y_ox, Z_ox, X_ctl, Y_ctl, Z_ctl = execute_cutoff(X_ctl, Y_ctl, Z_ctl, delta_s, critical_distance)
                # Re-sample center-line so that delta_s is roughly constant.
                X_ctl, Y_ctl, Z_ctl = resample_centerline(X_ctl, Y_ctl, Z_ctl, delta_s)
                # Save oxbow-lake parameters.
                oxbow_per_channel.append(Oxbow(X_ox, Y_ox, Z_ox, W, D))
                # Save river parameters.
                if it > 0 and it % save_it == 0:
                    centerline.append(Channel(X_ctl, Y_ctl, Z_ctl, W, D))
                # Print progress.
                sys.stdout.write('\rMigration progress:%.2f%%' % ((it + 1) / nit * 100))
            sys.stdout.write('\n')
            self.channel.append(centerline)  # Save different rivers.
            self.oxbow.append(oxbow_per_channel)  # Save oxbows of different rivers.
            # Show river migration and oxbows on X-Y plane.
            if show_migration:
                plot_channel2D(centerline, oxbow_per_channel, title=figure_title, interval=2)
            n_centerline = len(centerline)  # Number of saved rivers during migration.
            if mode == 'simple':  # In this mode, only simulate lag deposit of the most recent river.
                # Check requirement.
                if self.X_resolution != self.Y_resolution:
                    raise ValueError("For river deposits simulation, model's X & Y resolution must be identical.")
                # Get coordinates of the most recent center-line.
                X_ctl, Y_ctl, Z_ctl = centerline[-1].x, centerline[-1].y, centerline[-1].z
                # Select the center-line segment in target area.
                ind = (X_ctl >= X_pos - patch / 2) & (X_ctl <= X_pos + self.X_max - self.X_min + patch / 2)
                X_ctl, Y_ctl, Z_ctl = X_ctl[ind], Y_ctl[ind], Z_ctl[ind]
                # Re-sample center-line according to model's resolution.
                if delta_s > self.X_resolution:  # Center-line's resolution can not be larger than model's
                    X_ctl, Y_ctl, Z_ctl = resample_centerline(X_ctl, Y_ctl, Z_ctl, self.X_resolution)
                # Rotate the channel by its strike.
                R = np.array([[math.cos(math.radians(strike)), -math.sin(math.radians(strike))],  # Rotation matrix.
                              [math.sin(math.radians(strike)), math.cos(math.radians(strike))]], dtype='float32')
                center_X, center_Y = (self.X_max - self.X_min) / 2 + X_pos, Y_pos  # Rotation center.
                [X_ctl, Y_ctl] = R @ [X_ctl - center_X, Y_ctl - center_Y]  # Rotate.
                X_ctl += center_X
                Y_ctl += center_Y
                # Rasterize center-line and compute distance to center-line on X-Y plane.
                dist = compute_centerline_distance(X_ctl, Y_ctl, X_pos, self.X_min, self.Y_min, self.X_resolution,
                                                   self.X_points, self.Y_points)
                # Initialize topography.
                topo = np.ones(shape=[self.X_points, self.Y_points], dtype='float32') * Z_pos
                # River erosion.
                ze = erosion_surface(cl_dist=dist, z=Z_ctl, W=W, D=D)
                ze[ze < self.Z_min] = self.Z_min  # Limit in model's z-range.
                ze[ze > self.Z_max] = self.Z_max  # Limit in model's z-range.
                topo = np.maximum(topo, ze)
                # Riverbed lag deposit (gravel/conglomerate).
                zl = lag_surface(cl_dist=dist, z=Z_ctl, h_lag=h_lag, D=D)
                zl[zl < self.Z_min] = self.Z_min  # Limit in model's z-range.
                zl[zl > self.Z_max] = self.Z_max  # Limit in model's z-range.
                index = np.argwhere(zl < topo)
                indx, indy = index[:, 0], index[:, 1]
                indz1 = ((zl[indx, indy] - self.Z_min) / self.Z_resolution).astype('int32')
                indz2 = ((topo[indx, indy] - self.Z_min) / self.Z_resolution).astype('int32')
                for i in range(len(indx)):
                    sys.stdout.write('\rDeposition progress: %.2f%%' % ((i+1) / len(indx) * 100))
                    # Riverbed lag deposit: face code "1".
                    self.channel_facies[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 1
                    # Gravel: lithology code "1".
                    self.lith_facies[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 1
                    # Change marker.
                    marker[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 1
                sys.stdout.write('\n')
            if mode == 'complex':  # In this mode, simulate various kinds of river deposit.
                # Check requirement.
                if self.X_resolution != self.Y_resolution:
                    raise ValueError("For river deposits simulation, model's X & Y resolution must be identical.")
                # Initialize topography.
                topo = np.ones(shape=[self.X_points, self.Y_points], dtype='float32') * Z_pos
                # Initialize global oxbow-lake center-line distance with a large number.
                oxbow_dist = np.ones(shape=[self.X_points, self.Y_points], dtype='float32') * 1e10
                # Create river deposits center-line by center-line.
                for i_ctl in range(n_centerline):
                    # Get the coordinates of the center-line.
                    X_ctl, Y_ctl, Z_ctl = centerline[i_ctl].x, centerline[i_ctl].y, centerline[i_ctl].z
                    # Select the center-line segment in target area.
                    ind = (X_ctl >= X_pos - patch / 2) & (X_ctl <= X_pos + self.X_max - self.X_min + patch / 2)
                    X_ctl, Y_ctl, Z_ctl = X_ctl[ind], Y_ctl[ind], Z_ctl[ind]
                    # Re-sample center-line according to model's resolution.
                    if delta_s > self.X_resolution:  # Center-line's resolution must be smaller than model's.
                        X_ctl, Y_ctl, Z_ctl = resample_centerline(X_ctl, Y_ctl, Z_ctl, self.X_resolution)
                    # Rotate the river center-line by its strike.
                    R = np.array([[math.cos(math.radians(strike)), -math.sin(math.radians(strike))],  # Rotation matrix.
                                  [math.sin(math.radians(strike)), math.cos(math.radians(strike))]], dtype='float32')
                    center_X, center_Y = (self.X_max - self.X_min) / 2 + X_pos, Y_pos  # Rotation center.
                    [X_ctl, Y_ctl] = R @ [X_ctl - center_X, Y_ctl - center_Y]  # Rotate channel center-line.
                    X_ctl += center_X
                    Y_ctl += center_Y
                    # Rasterize channel center-line and compute distance to center-line on X-Y plane.
                    dist = compute_centerline_distance(X_ctl, Y_ctl, X_pos, self.X_min, self.Y_min, self.X_resolution,
                                                       self.X_points, self.Y_points)
                    # Make a list of oxbow-lake center-line coordinates between i and i-1 migration.
                    X_ox, Y_ox, Z_ox = [], [], []
                    # Check if cutoff happens before.
                    if i_ctl > 0:
                        for i_cn in range((i_ctl - 1) * save_it, i_ctl * save_it, 1):
                            X_oxbow, Y_oxbow, Z_oxbow = oxbow_per_channel[i_cn].x, oxbow_per_channel[i_cn].y, \
                                                        oxbow_per_channel[i_cn].z
                            if len(X_oxbow) > 0:
                                n_oxbow = len(X_oxbow)
                                for i_oxbow in range(n_oxbow):
                                    # Ensure that oxbow center-line's resolution is not larger than model's resolution.
                                    if delta_s > self.X_resolution:
                                        X_oxbow[i_oxbow], Y_oxbow[i_oxbow], Z_oxbow[i_oxbow] = \
                                            resample_centerline(X_oxbow[i_oxbow], Y_oxbow[i_oxbow],
                                                                Z_oxbow[i_oxbow], self.X_resolution)
                                    # Select oxbow center-line in target area.
                                    ind = (X_oxbow[i_oxbow] >= X_pos - patch / 2) & \
                                          (X_oxbow[i_oxbow] <= X_pos + self.X_max - self.X_min + patch / 2)
                                    X_oxbow[i_oxbow], Y_oxbow[i_oxbow], Z_oxbow[i_oxbow] = \
                                        X_oxbow[i_oxbow][ind], Y_oxbow[i_oxbow][ind], Z_oxbow[i_oxbow][ind]
                                    # Rotate oxbow-lake center-line by river strike.
                                    if len(X_oxbow[i_oxbow]) > 0:
                                        R = np.array([[math.cos(math.radians(strike)), -math.sin(math.radians(strike))],
                                                      [math.sin(math.radians(strike)), math.cos(math.radians(strike))]],
                                                     dtype='float32')  # Rotation matrix.
                                        # Rotation center.
                                        center_X, center_Y = (self.X_max - self.X_min) / 2 + X_pos, Y_pos
                                        # Rotate.
                                        [X_oxbow[i_oxbow], Y_oxbow[i_oxbow]] = \
                                            R @ [X_oxbow[i_oxbow] - center_X, Y_oxbow[i_oxbow] - center_Y]
                                        X_oxbow[i_oxbow] += center_X
                                        Y_oxbow[i_oxbow] += center_Y
                                        # Assemble the oxbows' coordinates between i and i-1 migration.
                                        X_ox.append(X_oxbow[i_oxbow])
                                        Y_ox.append(Y_oxbow[i_oxbow])
                                        Z_ox.append(Z_oxbow[i_oxbow])
                    # If cutoffs occur before, compute distance from their center-line.
                    if len(X_ox) > 0:
                        for i_ox in range(len(X_ox)):
                            # Compute distance from oxbow-lake center-line.
                            ox_dist = compute_centerline_distance(X_ox[i_ox], Y_ox[i_ox], X_pos, self.X_min,
                                                                  self.Y_min, self.X_resolution, self.X_points,
                                                                  self.Y_points)
                            # Update global oxbow-lake center-line distance.
                            oxbow_dist = np.minimum(oxbow_dist, ox_dist)
                            # Oxbow-lake erosion.
                            ze = erosion_surface(cl_dist=ox_dist, z=Z_ox[i_ox], W=W, D=D)
                            ze[ze < self.Z_min] = self.Z_min  # Limit in model's z-range.
                            ze[ze > self.Z_max] = self.Z_max  # Limit in model's z-range.
                            index = np.argwhere(ze > topo)
                            indx, indy = index[:, 0], index[:, 1]
                            indz1 = ((topo[indx, indy] - self.Z_min) / self.Z_resolution).astype('int32')
                            indz2 = ((ze[indx, indy] - self.Z_min) / self.Z_resolution).astype('int32')
                            for i in range(len(indx)):
                                # Other deposit: face code "0".
                                self.channel_facies[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 0
                                # Other lithology: lithology code "0".
                                self.lith_facies[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 0
                                # Change marker.
                                marker[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 0
                            topo = np.maximum(topo, ze)
                            # Oxbow-lake lag deposit (gravel/conglomerate).
                            zl = lag_surface(cl_dist=ox_dist, z=Z_ox[i_ox], h_lag=h_lag, D=D)
                            zl[ox_dist >= W] = 1e10  # Deposit inside oxbow.
                            zl[zl < self.Z_min] = self.Z_min  # Limit in model's z-range.
                            zl[zl > self.Z_max] = self.Z_max  # Limit in model's z-range.
                            index = np.argwhere(zl < topo)
                            indx, indy = index[:, 0], index[:, 1]
                            indz1 = ((zl[indx, indy] - self.Z_min) / self.Z_resolution).astype('int32')
                            indz2 = ((topo[indx, indy] - self.Z_min) / self.Z_resolution).astype('int32')
                            for i in range(len(indx)):
                                # Oxbow-lake deposit: face code "4".
                                self.channel_facies[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 4
                                # Gravel: lithology code "1".
                                self.lith_facies[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 1
                                # Change marker.
                                marker[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 1
                            topo = np.minimum(topo, zl)
                            # Oxbow-lake lag deposit (mudstone).
                            zl = lag_surface(cl_dist=ox_dist, z=Z_ox[i_ox], h_lag=h_lag + h_oxbow_mud, D=D)
                            zl[ox_dist >= W] = 1e10  # Deposit inside oxbow.
                            zl[zl < self.Z_min] = self.Z_min  # Limit in model's z-range.
                            zl[zl > self.Z_max] = self.Z_max  # Limit in model's z-range.
                            index = np.argwhere(zl < topo)
                            indx, indy = index[:, 0], index[:, 1]
                            indz1 = ((zl[indx, indy] - self.Z_min) / self.Z_resolution).astype('int32')
                            indz2 = ((topo[indx, indy] - self.Z_min) / self.Z_resolution).astype('int32')
                            for i in range(len(indx)):
                                # Oxbow-lake deposit: face code "4".
                                self.channel_facies[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 4
                                # Mudstone: lithology code "4"
                                self.lith_facies[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 4
                                # Change marker.
                                marker[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 4
                            topo = np.minimum(topo, zl)
                    # River erosion.
                    ze = erosion_surface(cl_dist=dist, z=Z_ctl, W=W, D=D)
                    ze[ze < self.Z_min] = self.Z_min  # Limit in model's z-range.
                    ze[ze > self.Z_max] = self.Z_max  # Limit in model's z-range.
                    index = np.argwhere(ze > topo)
                    indx, indy = index[:, 0], index[:, 1]
                    indz1 = ((topo[indx, indy] - self.Z_min) / self.Z_resolution).astype('int32')
                    indz2 = ((ze[indx, indy] - self.Z_min) / self.Z_resolution).astype('int32')
                    for i in range(len(indx)):
                        # Other deposit: face code "0".
                        self.channel_facies[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 0
                        # Other lithology: lithology code "0".
                        self.lith_facies[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 0
                        # Change marker.
                        marker[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 0
                    topo = np.maximum(topo, ze)
                    # Riverbed lag deposit (gravel/conglomerate).
                    zl = lag_surface(cl_dist=dist, z=Z_ctl, h_lag=h_lag, D=D)
                    zl[dist >= W] = 1e10  # Deposit inside channel.
                    zl[zl < self.Z_min] = self.Z_min  # Limit in model's z-range.
                    zl[zl > self.Z_max] = self.Z_max  # Limit in model's z-range.
                    index = np.argwhere(zl < topo)
                    indx, indy = index[:, 0], index[:, 1]
                    indz1 = ((zl[indx, indy] - self.Z_min) / self.Z_resolution).astype('int32')
                    indz2 = ((topo[indx, indy] - self.Z_min) / self.Z_resolution).astype('int32')
                    for i in range(len(indx)):
                        # Riverbed lag deposit: face code "1".
                        self.channel_facies[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 1
                        # Gravel: lithology code "1".
                        self.lith_facies[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 1
                        # Change marker.
                        marker[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 1
                    topo = np.minimum(topo, zl)
                    # Riverbed point-bar deposit.
                    if i_ctl != n_centerline - 1:  # The most recent river has no point-bar deposit.
                        zpb = pointbar_surface(cl_dist=dist, z=Z_ctl, W=W, D=D)
                        zpb[oxbow_dist <= W] = 1e10  # Clear point-bar deposit inside oxbow lake.
                        zpb[zpb < self.Z_min] = self.Z_min  # Limit in model's z-range.
                        zpb[zpb > self.Z_max] = self.Z_max  # Limit in model's z-range.
                        index = np.argwhere(zpb < topo)
                        indx, indy = index[:, 0], index[:, 1]
                        indz1 = ((zpb[indx, indy] - self.Z_min) / self.Z_resolution).astype('int32')
                        indz2 = ((topo[indx, indy] - self.Z_min) / self.Z_resolution).astype('int32')
                        for i in range(len(indx)):
                            # Riverbed point-bar deposit: face code "2".
                            self.channel_facies[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 2
                            # Sandstone: lithology code "2".
                            self.lith_facies[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 2
                            # Change marker.
                            marker[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 2
                        topo = np.minimum(topo, zpb)
                    # Natural levee deposit.
                    if i_ctl != n_centerline - 1:  # The most recent river has no levee deposit.
                        zlv = levee_surface(cl_dist=dist, h_levee=h_levee, w_levee=w_levee, W=W, tp=topo)
                        zlv[zlv < self.Z_min] = self.Z_min  # Limit in model's z-range.
                        zlv[zlv > self.Z_max] = self.Z_max  # Limit in model's z-range.
                        index = np.argwhere(zlv < topo)
                        indx, indy = index[:, 0], index[:, 1]
                        indz1 = ((zlv[indx, indy] - self.Z_min) / self.Z_resolution).astype('int32')
                        indz2 = ((topo[indx, indy] - self.Z_min) / self.Z_resolution).astype('int32')
                        for i in range(len(indx)):
                            # Natural levee deposit: face code "3".
                            self.channel_facies[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 3
                            # Siltstone: lithology code "3".
                            self.lith_facies[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 3
                            # Change marker.
                            marker[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 3
                        topo = np.minimum(topo, zlv)
                    # Print progress.
                    sys.stdout.write('\rDeposition progress:%.2f%%  centerline[%d/%d]' %
                                     ((i_ctl + 1) / n_centerline * 100, i_ctl + 1, n_centerline))
                sys.stdout.write('\n')
            # Update lithology parameters.
            gravel = self.lith_param.loc['conglomerate', :]  # Gravel parameters.
            sand = self.lith_param.loc['pebbled sandstone', :]  # Sandstone parameters.
            silt = self.lith_param.loc['siltstone', :]  # Siltstone parameters.
            mud = self.lith_param.loc['mudstone', :]  # Mudstone parameters.
            self.vp[(self.lith_facies == 1) & (marker == 1)] = random.uniform(gravel.vp_miu - gravel.vp_sigma,
                                                                              gravel.vp_miu + gravel.vp_sigma)
            self.vp[(self.lith_facies == 2) & (marker == 2)] = random.uniform(sand.vp_miu - sand.vp_sigma,
                                                                              sand.vp_miu + sand.vp_sigma)
            self.vp[(self.lith_facies == 3) & (marker == 3)] = random.uniform(silt.vp_miu - silt.vp_sigma,
                                                                              silt.vp_miu + silt.vp_sigma)
            self.vp[(self.lith_facies == 4) & (marker == 4)] = random.uniform(mud.vp_miu - mud.vp_sigma,
                                                                              mud.vp_miu + mud.vp_sigma)
            self.vs[(self.lith_facies == 1) & (marker == 1)] = random.uniform(gravel.vs_miu - gravel.vs_sigma,
                                                                              gravel.vs_miu + gravel.vs_sigma)
            self.vs[(self.lith_facies == 2) & (marker == 2)] = random.uniform(sand.vs_miu - sand.vs_sigma,
                                                                              sand.vs_miu + sand.vs_sigma)
            self.vs[(self.lith_facies == 3) & (marker == 3)] = random.uniform(silt.vs_miu - silt.vs_sigma,
                                                                              silt.vs_miu + silt.vs_sigma)
            self.vs[(self.lith_facies == 4) & (marker == 4)] = random.uniform(mud.vs_miu - mud.vs_sigma,
                                                                              mud.vs_miu + mud.vs_sigma)
            self.density[(self.lith_facies == 1) & (marker == 1)] = random.uniform(gravel.rho_miu - gravel.rho_sigma,
                                                                                   gravel.rho_miu + gravel.rho_sigma)
            self.density[(self.lith_facies == 2) & (marker == 2)] = random.uniform(sand.rho_miu - sand.rho_sigma,
                                                                                   sand.rho_miu + sand.rho_sigma)
            self.density[(self.lith_facies == 3) & (marker == 3)] = random.uniform(silt.rho_miu - silt.rho_sigma,
                                                                                   silt.rho_miu + silt.rho_sigma)
            self.density[(self.lith_facies == 4) & (marker == 4)] = random.uniform(mud.rho_miu - mud.rho_sigma,
                                                                                   mud.rho_miu + mud.rho_sigma)
        # Print simulation time.
        t_end = time.perf_counter()
        print('Simulation time: %.2fs' % (t_end - t_start))

    def rectangular_grid(self, resolution=None, param=None, method='nearest', fill_value=np.nan):
        """
        Re-sample the model on rectangular (quad) grid using interpolation.
        :param resolution: (List of floats) - Default is [4.0, 4.0, 4.0]. The 3D grid spacing.
        :param param: (String or list of strings) - Default is 'all'. Model parameters to be re-sampled.
                      If 'all', will re-sample all parameters,
                      or you can choose one or more parameters like ['vp'] or ['rc', 'vp', 'vs'].
        :param method: (String) - Default is 'nearest'. Method of interpolation.
                       Options are 'nearest', 'linear' and 'cubic'.
        :param fill_value: (Float) - Default is NaN.Value used to fill in for requested points outside of the
                           convex hull of the input points. This parameter has no effect for the 'nearest' method.
        """
        # Before interpolation, store the boundary coordinates for auto-cropping the model.
        self.autocrop = {'xmin': np.round(np.amax(self.X[:, 0, :]), 2), 'xmax': np.round(np.amin(self.X[:, -1, :]), 2),
                         'ymin': np.round(np.amax(self.Y[0, :, :]), 2), 'ymax': np.round(np.amin(self.Y[-1, :, :]), 2),
                         'zmin': np.round(np.amax(self.Z[:, :, 0]), 2), 'zmax': np.round(np.amin(self.Z[:, :, -1]), 2)}
        # Set resolution of 3D rectangular grid.
        if resolution is None:
            dx, dy, dz = self.X_resolution, self.Y_resolution, self.Z_resolution
        else:
            dx, dy, dz = resolution[0], resolution[1], resolution[2]
            self.X_resolution, self.Y_resolution, self.Z_resolution = dx, dy, dz
        print('Interpolate on %.2fm x %.2fm x %.2fm grid...' % (dx, dy, dz))
        # Make 3D rectangular grid.
        x = np.arange(start=self.X_min, stop=self.X_max + dx, step=dx)
        y = np.arange(start=self.Y_min, stop=self.Y_max + dy, step=dy)
        z = np.arange(start=self.Z_min, stop=self.Z_max + dz, step=dz)
        x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z)
        # Interpolate.
        t_start = time.perf_counter()
        ctp = np.c_[self.X.ravel(order='F'),
                    self.Y.ravel(order='F'),
                    self.Z.ravel(order='F')]  # Coordinates of control points.
        self.X, self.Y, self.Z = x_mesh.copy(), y_mesh.copy(), z_mesh.copy()
        self.Y_points, self.X_points, self.Z_points = self.X.shape
        if param is None:
            param = 'all'
        if param == 'all' or 'rc' in param:
            sys.stdout.write('Interpolating reflection coefficient...')
            self.rc = scipy.interpolate.griddata(points=ctp, values=self.rc.ravel(order='F'),
                                                 xi=(x_mesh, y_mesh, z_mesh),
                                                 method=method, fill_value=fill_value)  # Reflection coefficient.
            sys.stdout.write('Done.\n')
        if param == 'all' or 'vp' in param:
            sys.stdout.write('Interpolating p-wave velocity...')
            self.vp = scipy.interpolate.griddata(points=ctp, values=self.vp.ravel(order='F'),
                                                 xi=(x_mesh, y_mesh, z_mesh),
                                                 method=method, fill_value=fill_value)  # P-wave velocity.
            sys.stdout.write('Done.\n')
        if param == 'all' or 'vs' in param:
            sys.stdout.write('Interpolating s-wave velocity...')
            self.vs = scipy.interpolate.griddata(points=ctp, values=self.vs.ravel(order='F'),
                                                 xi=(x_mesh, y_mesh, z_mesh),
                                                 method=method, fill_value=fill_value)  # S-wave velocity.
            sys.stdout.write('Done.\n')
        if param == 'all' or 'density' in param:
            sys.stdout.write('Interpolating density...')
            self.density = scipy.interpolate.griddata(points=ctp, values=self.density.ravel(order='F'),
                                                      xi=(x_mesh, y_mesh, z_mesh),
                                                      method=method, fill_value=fill_value)  # density.
            sys.stdout.write('Done.\n')
        if param == 'all' or 'Ip' in param:
            sys.stdout.write('Interpolating p-wave impedance...')
            self.Ip = scipy.interpolate.griddata(points=ctp, values=self.Ip.ravel(order='F'),
                                                 xi=(x_mesh, y_mesh, z_mesh),
                                                 method=method, fill_value=fill_value)  # P-wave impedance.
            sys.stdout.write('Done.\n')
        if param == 'all' or 'Is' in param:
            sys.stdout.write('Interpolating s-wave impedance...')
            self.Is = scipy.interpolate.griddata(points=ctp, values=self.Is.ravel(order='F'),
                                                 xi=(x_mesh, y_mesh, z_mesh),
                                                 method=method, fill_value=fill_value)  # S-wave impedance.
            sys.stdout.write('Done.\n')
        if param == 'all' or 'lith_facies' in param:
            sys.stdout.write('Interpolating lithology facies...')
            self.lith_facies = scipy.interpolate.griddata(points=ctp, values=self.lith_facies.ravel(order='F'),
                                                          xi=(x_mesh, y_mesh, z_mesh),
                                                          method=method, fill_value=fill_value)  # Lithology facies.
            sys.stdout.write('Done.\n')
        if param == 'all' or 'channel_facies' in param:
            sys.stdout.write('Interpolating channel facies...')
            self.channel_facies = scipy.interpolate.griddata(points=ctp, values=self.channel_facies.ravel(order='F'),
                                                             xi=(x_mesh, y_mesh, z_mesh),
                                                             method=method, fill_value=fill_value)  # Channel facies.
            sys.stdout.write('Done.\n')
        if param == 'all' or 'fm' in param:
            sys.stdout.write('Interpolating fault marker...')
            self.fm = scipy.interpolate.griddata(points=ctp, values=self.fm.ravel(order='F'),
                                                 xi=(x_mesh, y_mesh, z_mesh),
                                                 method=method, fill_value=fill_value)  # Fault marker.
            sys.stdout.write('Done.\n')
        t_end = time.perf_counter()
        print('Interpolation finished. Time: %.2fs' % (t_end - t_start))
        print('Model extent:')
        print('X range: %.2fm-%.2fm' % (self.X_min, self.X_max))
        print('Y range: %.2fm-%.2fm' % (self.Y_min, self.Y_max))
        print('Z range: %.2fm-%.2fm' % (self.Z_min, self.Z_max))
        print('Model resolution (XYZ): [%.2fm x  %.2fm x %.2fm]' %
              (self.X_resolution, self.Y_resolution, self.Z_resolution))
        print('Model points (XYZ): [%d x %d x %d]' % (self.X_points, self.Y_points, self.Z_points))

    def crop(self, x_bound=None, y_bound=None, z_bound=None, param=None):
        """
        Crop rectangular (quad) grid model.
        :param x_bound: (List of floats) - X coordinates used to crop the model, which is a list of [xmin, xmax].
                        For example, x_bound = [200, 1000] means to crop the model from x = 200 to x = 1000.
                        If x_bound is None, will automatically crop the model from the maximum x of the east boundary to
                        the minimum x of the west boundary.
        :param y_bound: (List of floats) - Y coordinates used to crop the model, which is a list of [ymin, ymax].
                        For example, y_bound = [200, 1000] means to crop the model from y = 200 to y = 1000.
                        If y_bound is None, will automatically crop the model from the maximum y of the south boundary
                        to the minimum y of the north boundary.
        :param z_bound: (List of floats) - Z coordinates used to crop the model, which is a list of [zmin, zmax].
                        For example, z_bound = [200, 1000] means to crop the model from z = 200 to z = 1000.
                        If z_bound is None, will automatically crop the model from the maximum z of the bottom boundary
                        to the minimum z of the top boundary.
        :param param: (String or list of strings) - Default is 'all'. The parameter cube to crop.
                      If 'all', will crop all parameter cubes.
                      Or you can choose one or more parameter cubes like ['vp'] or ['vp', 'Ip', 'rc']
        """
        print('Cropping model...')
        if x_bound is None:  # When x_bound is None, auto-crop the model in x direction.
            condx = (self.X >= self.autocrop['xmin']) & (self.X <= self.autocrop['xmax'])
            print('Crop X from %.2fm-%.2fm (auto-crop)' % (self.autocrop['xmin'], self.autocrop['xmax']))
        else:  # When x_bound is defined, crop the model by defined x coordinates
            condx = (self.X >= min(x_bound)) & (self.X <= max(x_bound))
            print('Crop X from %.2fm-%.2fm' % (min(x_bound), max(x_bound)))
        if y_bound is None:  # When y_bound is None, auto-crop the model in y direction.
            condy = (self.Y >= self.autocrop['ymin']) & (self.Y <= self.autocrop['ymax'])
            print('Crop Y from %.2fm-%.2fm (auto-crop)' % (self.autocrop['ymin'], self.autocrop['ymax']))
        else:  # When y_bound is defined, crop the model by defined y direction.
            condy = (self.Y >= min(y_bound)) & (self.Y <= max(y_bound))
            print('Crop Y from %.2fm-%.2fm' % (min(y_bound), max(y_bound)))
        if z_bound is None:  # When z_bound is None, auto-crop the model in z direction.
            condz = (self.Z >= self.autocrop['zmin']) & (self.Z <= self.autocrop['zmax'])
            print('Crop Z from %.2fm-%.2fm (auto-crop)' % (self.autocrop['zmin'], self.autocrop['zmax']))
        else:  # When z_bound is defined, crop the model by defined z coordinates.
            condz = (self.Z >= min(z_bound)) & (self.Z <= max(z_bound))
            print('Crop Z from %.2fm-%.2fm (auto-crop)' % (min(z_bound), max(z_bound)))
        if param is None:
            param = 'all'
        # XYZ indexes that meet the conditions.
        indx = np.argwhere(condx)[:, 1]
        indy = np.argwhere(condy)[:, 0]
        indz = np.argwhere(condz)[:, -1]
        # Cropping the model.
        sys.stdout.write('Cropping X cube...')
        self.X = self.X[min(indy):max(indy) + 1, min(indx):max(indx) + 1, min(indz):max(indz) + 1]
        sys.stdout.write('Done.\n')
        sys.stdout.write('Cropping Y cube...')
        self.Y = self.Y[min(indy):max(indy) + 1, min(indx):max(indx) + 1, min(indz):max(indz) + 1]
        sys.stdout.write('Done.\n')
        sys.stdout.write('Cropping Z cube...')
        self.Z = self.Z[min(indy):max(indy) + 1, min(indx):max(indx) + 1, min(indz):max(indz) + 1]
        sys.stdout.write('Done.\n')
        if param == 'all' or 'rc' in param:
            sys.stdout.write('Cropping rc cube...')
            self.rc = self.rc[min(indy):max(indy) + 1, min(indx):max(indx) + 1, min(indz):max(indz) + 1]
            sys.stdout.write('Done.\n')
        if param == 'all' or 'vp' in param:
            sys.stdout.write('Cropping vp cube...')
            self.vp = self.vp[min(indy):max(indy) + 1, min(indx):max(indx) + 1, min(indz):max(indz) + 1]
            sys.stdout.write('Done.\n')
        if param == 'all' or 'vs' in param:
            sys.stdout.write('Cropping vs cube...')
            self.vs = self.vs[min(indy):max(indy) + 1, min(indx):max(indx) + 1, min(indz):max(indz) + 1]
            sys.stdout.write('Done.\n')
        if param == 'all' or 'density' in param:
            sys.stdout.write('Cropping density cube...')
            self.density = self.density[min(indy):max(indy) + 1, min(indx):max(indx) + 1, min(indz):max(indz) + 1]
            sys.stdout.write('Done.\n')
        if param == 'all' or 'Ip' in param:
            sys.stdout.write('Cropping Ip cube...')
            self.Ip = self.Ip[min(indy):max(indy) + 1, min(indx):max(indx) + 1, min(indz):max(indz) + 1]
            sys.stdout.write('Done.\n')
        if param == 'all' or 'Is' in param:
            sys.stdout.write('Cropping Is cube...')
            self.Is = self.Is[min(indy):max(indy) + 1, min(indx):max(indx) + 1, min(indz):max(indz) + 1]
            sys.stdout.write('Done.\n')
        if param == 'all' or 'seis' in param:
            sys.stdout.write('Cropping seis cube...')
            self.seis = self.seis[min(indy):max(indy) + 1, min(indx):max(indx) + 1, min(indz):max(indz) + 1]
            sys.stdout.write('Done.\n')
        if param == 'all' or 'lith_facies' in param:
            sys.stdout.write('Cropping lith_facies cube...')
            self.lith_facies = self.lith_facies[min(indy):max(indy) + 1, min(indx):max(indx) + 1,
                                                min(indz):max(indz) + 1]
            sys.stdout.write('Done.\n')
        if param == 'all' or 'channel_facies' in param:
            sys.stdout.write('Cropping channel_facies cube...')
            self.channel_facies = self.channel_facies[min(indy):max(indy) + 1, min(indx):max(indx) + 1,
                                                      min(indz):max(indz) + 1]
            sys.stdout.write('Done.\n')
        if param == 'all' or 'fm' in param:
            sys.stdout.write('Cropping rc cube...')
            self.fm = self.fm[min(indy):max(indy) + 1, min(indx):max(indx) + 1, min(indz):max(indz) + 1]
            sys.stdout.write('Done.\n')
        self.X_min, self.X_max = np.amin(self.X), np.amax(self.X)
        self.Y_min, self.Y_max = np.amin(self.Y), np.amax(self.Y)
        self.Z_min, self.Z_max = np.amin(self.Z), np.amax(self.Z)
        self.Y_points, self.X_points, self.Z_points = self.X.shape
        # Print model info.
        print('Model extent:')
        print('X range: %.2fm-%.2fm' % (self.X_min, self.X_max))
        print('Y range: %.2fm-%.2fm' % (self.Y_min, self.Y_max))
        print('Z range: %.2fm-%.2fm' % (self.Z_min, self.Z_max))
        print('Model resolution (XYZ): [%.2fm x %.2fm x %.2fm]' %
              (self.X_resolution, self.Y_resolution, self.Z_resolution))
        print('Model points (XYZ): [%d x %d x %d]' % (self.X_points, self.Y_points, self.Z_points))

    def show(self, plotter=None, param=None, zscale=None, cmap=None, slices=False, point_cloud=False,
             hide_value=None):
        """
        Visualize the model.
        :param plotter: (pyvista.Plotter) - Default is None, which is to create a new plotter. Can also accept a plotter
                        from outside.
        :param param: (String) - Choose a parameter to visualize.
                      Options are 'rc', 'vp', 'vs', 'density', 'channel_facies', 'lith_facies' and 'fm'.
        :param zscale: (Float or string) - Scaling in the z direction. Default is not to change the existing scaling.
                       If 'auto', will scale the min(x range, y range) / z range to the Golden Ratio.
        :param cmap: (String) - Colormap used to visualize the model.
        :param slices: (Bool) - Whether to display the model as orthogonal slices. Default is False.
        :param point_cloud: (Bool) - Whether to display the model as point cloud. Default is False.
        :param hide_value: (List) - List of values that will be hidden when visualizing the model as point cloud.
                           For example, hide_value=[0, 2, 3] means not to display points with value=0, 2 or 3.
                           Only be effective when point_cloud=True.
        :return: plotter: (pyvista.Plotter) - A pyvista plotter with the visualized model.
        """
        if plotter is None:
            pv.set_plot_theme('document')
            plotter = BackgroundPlotter()
        if zscale is not None:
            if zscale == 'auto':
                zscale = min(self.X_max - self.X_min, self.Y_max - self.Y_min) / (self.Z_max - self.Z_min) * 0.618
            plotter.set_scale(zscale=zscale)
        if cmap is None:
            cmap = 'viridis'
        points = np.c_[self.X.ravel(order='F'), self.Y.ravel(order='F'), self.Z.ravel(order='F')]
        grid = pv.StructuredGrid(self.X, self.Y, self.Z)
        if param == 'rc':
            values = self.rc.ravel(order='F')
            scalar_bar_title = 'RC'
            title = 'Reflection Coefficient Model'
        if param == 'vp':
            values = self.vp.ravel(order='F')
            scalar_bar_title = 'Vp(km/s)'
            title = 'P-wave Velocity Model'
        if param == 'vs':
            values = self.vs.ravel(order='F')
            scalar_bar_title = 'Vs(km/s)'
            title = 'S-wave Velocity Model'
        if param == 'density':
            values = self.density.ravel(order='F')
            scalar_bar_title = 'Density(g/cm^3)'
            title = 'Density Model'
        if param == 'Ip':
            values = self.Ip.ravel(order='F')
            scalar_bar_title = 'Ip'
            title = 'P-wave Impedance'
        if param == 'Is':
            values = self.Is.ravel(order='F')
            scalar_bar_title = 'Is'
            title = 'S-wave Impedance'
        if param == 'seis':
            values = self.seis.ravel(order='F')
            scalar_bar_title = 'amp'
            title = 'Seismic'
        if param == 'channel_facies':
            values = self.channel_facies.ravel(order='F')
            scalar_bar_title = 'Channel Facies Code'
            title = 'Channel Facies Model'
        if param == 'lith_facies':
            values = self.lith_facies.ravel(order='F')
            scalar_bar_title = 'Lithology Facies Code'
            title = 'Lithology Facies Model'
        if param == 'fm':
            values = self.fm.ravel(order='F')
            scalar_bar_title = 'Fault Probability'
            title = 'Fault Probability Model'
        sargs = dict(height=0.5, vertical=True, position_x=0.85, position_y=0.05,
                     title=scalar_bar_title, title_font_size=20)
        if point_cloud:
            if hide_value is not None:
                for v in hide_value:
                    points = points[values != v, :]
                    values = values[values != v]
            pc = pv.PolyData(points)
            pc[param] = values
            plotter.add_mesh(pc, render_points_as_spheres=True, scalars=param, show_scalar_bar=True, cmap=cmap,
                             scalar_bar_args=sargs)
        else:
            grid[param] = values
            if slices:
                plotter.add_mesh_slice_orthogonal(grid, scalars=param, show_scalar_bar=True, cmap=cmap,
                                                  scalar_bar_args=sargs)
            else:
                plotter.add_mesh(grid, scalars=param, show_scalar_bar=True, cmap=cmap, scalar_bar_args=sargs)
        plotter.add_mesh(grid.outline(), color='k')
        plotter.add_text(title, font_size=15)
        plotter.show_bounds()
        plotter.add_axes()
        return plotter


def compute_f_parallel(i, x=None, y=None, interpolator=None):
    """
    Compute the curved fault surface's z coordinates using the bi-harmonic spline interpolation in parallel .
    :param i: (Integer) - The slice index number (axis=0) of the model's x and y coordinate arrays in fault plane
              coordinate system.
    :param x: (numpy.3darray) - The model's x coordinate array in fault plane coordinate system.
    :param y: (numpy.3darray) - The model's y coordinate array in fault plane coordinate system.
    :param interpolator: (class BiharmonicSpline3D) - The bi-harmonic spline interpolator initialized by
                         random perturbation points near the planar fault plane.
    :return: (List of numpy.2darrays) - A slice (axis=0) of curved fault surface's z coordinates in fault plane
             coordinate system.
    """
    out = interpolator(x[i, :, :], y[i, :, :])
    return out


def compute_Dz_parallel(i, x=None, y=None, f=None, interpolator=None):
    """
    Compute the model's displacement in the fault surface's normal direction.
    :param i: (Integer) - The slice index number (axis=0) of the model's x and y coordinate arrays in fault plane
              coordinate system.
    :param x: (numpy.3darray) - The model's x coordinate array in fault plane coordinate system.
    :param y: (numpy.3darray) - The model's y coordinate array in fault plane coordinate system.
    :param f: (numpy.3darray) - The fault surface's z-coordinate array in fault plane coordinate system.
    :param interpolator: (class BiharmonicSpline3D) - The bi-harmonic spline interpolator initialized by
                         random perturbation points near the planar fault plane.
    :return: (List of numpy.2darrays) - A slice (axis=0) of the model's displacement in the fault plane's
             normal direction.
    """
    out = interpolator(x[i, :, :], y[i, :, :]) - f[i, :, :]
    return out


class Channel:
    """
    Store the river center-line coordinates, river width and maximum depth.
    """

    def __init__(self, x, y, z, W, D):
        """
        :param x: (numpy.1darray) - x-coordinates of center-line.
        :param y: (numpy.1darray) - y-coordinates of center-line.
        :param z: (numpy.1darray) - z-coordinates of center-line.
        :param W: (Float) - River width.
        :param D: (Float) - River maximum depth.
        """
        self.x = x
        self.y = y
        self.z = z
        self.W = W
        self.D = D


class Oxbow:
    """
    Store the oxbow-lake center-line coordinates, oxbow-lake width and maximum depth.
    """

    def __init__(self, xc, yc, zc, W, D):
        """
        :param xc: (numpy.1darray) - x-coordinates of oxbow center-line.
        :param yc: (numpy.1darray) - y-coordinates of oxbow center-line.
        :param zc: (numpy.1darray) - z-coordinates of oxbow center-line.
        :param W: (Float) - Oxbow-lake width.
        :param D: (Float) - Oxbow-lake maximum depth.
        """
        self.x = xc
        self.y = yc
        self.z = zc
        self.W = W
        self.D = D


def initialize_centerline(s_init, ypos, zpos, delta_s, n_bends, perturb):
    """
    Initialize river center-line. Assuming x is the longitudinal flow direction. First create a straight river
    center-line, then add perturbation to make it bended.
    :param s_init: (Float) - Length of the straight center-line.
    :param ypos: (Float) - y position of the center-line.
    :param zpos: (Float) - z position of the center-line.
    :param delta_s: (Float) - Distance between two consecutive points along center-line.
    :param n_bends: (Integer) - Number of bends in the center-line.
    :param perturb: (List) - y-coordinates' range of perturbation points.
    :return: x: (numpy.1darray) - x-coordinates of the initial center-line.
             y: (numpy.1darray) - y-coordinates of the initial center-line.
             z: (numpy.1darray) - z-coordinates of the initial center-line.
    """
    x = np.arange(0, s_init + delta_s, delta_s, dtype='float32')
    # Generate perturbation points.
    xp = np.linspace(0, s_init, n_bends + 2, dtype='float32')
    yp = np.ones(len(xp), dtype='float32') * ypos
    for i in range(1, len(yp) - 1):
        ptb = random.uniform(perturb[0], perturb[1])
        yp[i] += (-1) ** i * ptb
    # Interpolate bended center-line.
    interpolator = BiharmonicSpline2D(xp, yp)
    y = interpolator(x)
    z = np.ones(len(x), dtype='float32') * zpos
    return x, y, z


def resample_centerline(x, y, z, delta_s):
    """
    Re-sample center-line so that delta_s is roughly constant. Modified from Zoltan Sylvester's meanderpy.
    [https://github.com/zsylvester/meanderpy]
    :param x: (numpy.1darray) - x-coordinates of center-line.
    :param y: (numpy.1darray) - y-coordinates of center-line.
    :param z: (numpy.1darray) - z-coordinates of center-line.
    :param delta_s: (Float) - Distance between two consecutive points along center-line.
    :return:
    """
    dx, dy, s, ds = compute_curvelength(x, y)
    # Cubic spline interpolation. s=0 means no smoothing.
    tck = scipy.interpolate.splprep([x, y, z], s=0)
    unew = np.linspace(0, 1, 1 + int(round(s / delta_s)))
    out = scipy.interpolate.splev(unew, tck[0])
    x_res, y_res, z_res = out[0], out[1], out[2]
    x_res, y_res, z_res = x_res.astype('float32'), y_res.astype('float32'), z_res.astype('float32')
    return x_res, y_res, z_res


def compute_curvelength(x, y):
    """
    Compute the length of center-line. Modified from Zoltan Sylvester's meanderpy.
    [https://github.com/zsylvester/meanderpy]
    :param x: (numpy.1darray) - x-coordinates of center-line.
    :param y: (numpy.1darray) - y-coordinates of center-line.
    :return: dx: (numpy.1darray) - First derivative of each point's x-coordinates on center-line.
             dy: (numpy.1darray) - First derivative of each point's y-coordinates on center-line.
             s: (Float) - The length of center-line.
             ds: (numpy.1darray) - The length of curve between two consecutive points along the center-line.
    """
    dx = np.gradient(x)
    dy = np.gradient(y)
    ds = np.sqrt(dx ** 2 + dy ** 2)
    s = np.sum(ds[1:])
    return dx, dy, s, ds


def compute_curvature(x, y):
    """
    Compute the curvatures at each points of center-line. Modified from Zoltan Sylvester's meanderpy.
    [https://github.com/zsylvester/meanderpy]
    :param x: (numpy.1darray) - x-coordinates of center-line.
    :param y: (numpy.1darray) - y-coordinates of center-line.
    :return: curvature: (numpy.1darray) - Curvatures at each points of center-line.
    """
    # First derivatives.
    dx = np.gradient(x)
    dy = np.gradient(y)
    # Second derivatives.
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5)
    return curvature


@numba.jit(nopython=True)  # Use numba to speed up the computation.
def compute_tangential_angle(x, y):
    """
    Compute tangential angle at each point of center-line.
    :param x: (numpy.1darray) - x-coordinates of center-line.
    :param y: (numpy.1darray) - y-coordinates of center-line.
    :return: beta: (numpy.1darray) - Tangential angle (radian) of each point.
    """
    beta = np.zeros(len(x), dtype='float32')  # Initialization.
    for i in range(len(x)):
        # The first point.
        if i == 0:
            if x[i + 1] == x[i]:  # Avoid division by zero.
                beta[i] = math.atan((y[i + 1] - y[i]) / 1e-6)
            else:
                beta[i] = math.atan((y[i + 1] - y[i]) / (x[i + 1] - x[i]))
            # The arc-tangent function can only return [-90? 90i, which means the angle in first quadrant is the same
            # as the angle in third quadrant, and the angle in second quadrant is the same as the angle in fourth
            # quadrant. The angles are in [-180? 180i through the process below.
            if y[i + 1] > y[i] and x[i + 1] < x[i]:
                beta[i] += math.pi
            if y[i + 1] < y[i] and x[i + 1] < x[i]:
                beta[i] -= math.pi
        # The end point.
        elif i == len(x) - 1:
            if x[i] == x[i - 1]:  # Avoid division by zero.
                beta[i] = math.atan((y[i] - y[i - 1]) / 1e-6)
            else:
                beta[i] = math.atan((y[i] - y[i - 1]) / (x[i] - x[i - 1]))
            # Angle transform.
            if y[i] > y[i - 1] and x[i] < x[i - 1]:
                beta[i] += math.pi
            if y[i] < y[i - 1] and x[i] < x[i - 1]:
                beta[i] -= math.pi
        # The interval points. Use three points (backward and forward) to compute the tangential angle.
        else:
            if x[i + 1] == x[i]:  # Avoid division by zero.
                beta_forward = math.atan((y[i + 1] - y[i]) / 1e-6)
            else:
                beta_forward = math.atan((y[i + 1] - y[i]) / (x[i + 1] - x[i]))
            if x[i] == x[i - 1]:  # Avoid division by zero.
                beta_backward = math.atan((y[i] - y[i - 1]) / 1e-6)
            else:
                beta_backward = math.atan((y[i] - y[i - 1]) / (x[i] - x[i - 1]))
            # Angle transform.
            if y[i + 1] > y[i] and x[i + 1] < x[i]:
                beta_forward += math.pi
            if y[i + 1] < y[i] and x[i + 1] < x[i]:
                beta_forward -= math.pi
            if y[i] > y[i - 1] and x[i] < x[i - 1]:
                beta_backward += math.pi
            if y[i] < y[i - 1] and x[i] < x[i - 1]:
                beta_backward -= math.pi
            beta[i] = 0.5 * (beta_forward + beta_backward)
            # This is the situation that the flow direction is opposite to the x-direction AND the middle point is
            # higher or lower than both forward point and backward point.
            if x[i + 1] < x[i - 1] and \
                    ((y[i] >= y[i + 1] and y[i] >= y[i - 1]) or (y[i] <= y[i - 1] and y[i] <= y[i + 1])):
                if beta[i] >= 0.0:
                    beta[i] -= math.pi
                else:
                    beta[i] += math.pi
    return beta


@numba.jit(nopython=True)  # Use numba to speed up the computation.
def compute_migration_rate(curv, ds, W, kl, Cf, D, pad_up, pad_down):
    """
    Compute migration rate of Howard-Knutson (1984) model. Modified from Zoltan Sylvester's meanderpy.
    [https://github.com/zsylvester/meanderpy]
    :param curv: (numpy.1darray) - Curvature of each point on center-line.
    :param ds: (numpy.1darray) - Distances between two consecutive points on center-line.
    :param W: (Float) - River's width.
    :param kl: (Float) - Migration constant (m/year).
    :param Cf: (Float) - Friction factor.
    :param D: (Float) - River's depth.
    :param pad_up: (Integer) - Number of points that will not migrate at upstream.
    :param pad_down: (Integer) - Number of points that will not migrate at downstream.
    :return: R1: (numpy.1darray) - The migration rate.
    """
    omega = -1.0
    gamma = 2.5
    k = 1.0
    R0 = kl * W * curv  # Nominal migration rate.
    R1 = np.zeros(len(R0), dtype='float32')  # Initialize adjusted migration rate.
    alpha = 2 * k * Cf / D
    if pad_up < 5:
        pad_up = 5
    for i in range(pad_up, len(R0) - pad_down):
        si = np.concatenate(
            (np.array([0]), np.cumsum(ds[i - 1::-1])))  # Cumulate distances backward from current point.
        G = np.exp(-alpha * si)
        # Adjusted migration rate in Howard-Knutson model.
        R1[i] = omega * R0[i] + gamma * np.sum(R0[i::-1] * G) / np.sum(G)
    return R1


def channel_bank(x, y, W):
    """
    Compute river banks' coordinates.
    :param x: (numpy.1darray) - x-coordinates of the center-line.
    :param y: (numpy.1darray) - y-coordinates of the center-line.
    :param W: (Float) - The channel's width.
    :return: xb: (numpy.2darray) - The x-coordinates of river banks. Shape: [len(x), 2].
                 Each row is the x-coordinates of two banks of a point in center-line.
             yb: (numpy.2darray) - The x-coordinates of river banks. Shape: [len(x), 2].
                 Each row is the y-coordinates of two banks of a point in center-line.
    """
    ns = len(x)
    angle = compute_tangential_angle(x, y)
    # Get the parabolas' endpoints' y-coordinates of each points on center-line.
    # Note that this is not the bank's y-coordinates until they are rotated.
    xb = np.c_[x, x]
    yb = np.c_[y - W / 2, y + W / 2]
    # Compute the parabola.
    for i in range(ns):
        R = np.array([[math.cos(angle[i]), -math.sin(angle[i])],  # Rotation matrix
                      [math.sin(angle[i]), math.cos(angle[i])]])
        [xb[i, :], yb[i, :]] = R @ [xb[i, :] - x[i], yb[i, :] - y[i]]  # Rotate to normal direction.
        xb[i, :] += x[i]  # x-coordinates of the erosion surface.
        yb[i, :] += y[i]  # y-coordinates of the erosion surface.
    return xb, yb


def compute_centerline_distance(x, y, xpos, xmin, ymin, dx, nx, ny):
    """
    Rasterize center-line and compute distance to center-line on X-Y plane. Modified from Zoltan Sylvester's meanderpy.
    [https://github.com/zsylvester/meanderpy]
    :param x: (numpy.1darray) - x-coordinates of center-line.
    :param y: (numpy.1darray) - y-coordinates of center-line.
    :param xpos: (Float) - Center-line's x-coordinate which the 3D model starts at.
    :param xmin: (Float) - Minimum x-coordinates of the model.
    :param ymin: (Float) - Minimum y-coordinates of the model.
    :param dx: (Float) - X & Y resolution of the model.
    :param nx: (Integer) - Number of points on model's x-direction.
    :param ny: (Integer) - Number of points on model's y-direction.
    :return: dist: (numpy.2darray) - Distance to center-line on X-Y plane.
    """
    ctl_pixels = []
    offset = xpos - xmin
    for i in range(len(x)):
        px = int((x[i] - offset - xmin) / dx)
        py = int((y[i] - ymin) / dx)
        if 0 <= px < nx and 0 <= py < ny:
            ctl_pixels.append((py, px))
    # Rasterize center-line.
    img = Image.new(mode='RGB', size=(ny, nx), color='white')  # Background is white.
    draw = ImageDraw.Draw(img)
    draw.line(ctl_pixels, fill='rgb(0, 0, 0)')  # Center-line is black.
    # Transfer image to array.
    pix = np.array(img)
    ctl = pix[:, :, 0]
    ctl[ctl == 255] = 1  # Background is 1, center-line is 0.
    # Compute distance to center-line.
    dist_map = ndimage.distance_transform_edt(ctl)
    dist = dist_map * dx  # The real distance.
    dist.astype('float32')
    return dist


def erosion_surface(cl_dist, z, W, D):
    """
    Create erosion surface.
    :param cl_dist: (numpy.2darray) - Distance from center-line on X-Y plane.
    :param z: (numpy.1darray) - z-coordinates of center-line.
    :param W: (Float) - River's width.
    :param D: (Float) - River's maximum depth.
    :return: ze: (numpy.2darray) - z-coordinates of erosion surface.
    """
    if len(z[z - z[0] != 0]):
        raise ValueError('Can not process center-line with changing z-coordinates.')
    ze = z[0] + 4 * D / W ** 2 * (W ** 2 / 4 - cl_dist ** 2)
    ze = ze.astype('float32')
    return ze


def lag_surface(cl_dist, z, h_lag, D):
    """
    Create Riverbed lag deposit surface.
    :param cl_dist: (numpy.2darray) - Distance from center-line on X-Y plane.
    :param z: (numpy.1darray) - z-coordinates of center-line.
    :param h_lag: (Float) - The maximum thickness of lag deposit.
    :param D: (Float) - River's maximum depth.
    :return: zl: (numpy.2darray) - z-coordinates of lag deposit surface.
    """
    if len(z[z - z[0] != 0]):
        raise ValueError('Can not process center-line with changing z-coordinates.')
    zl = (z[0] + D - h_lag) * np.ones(shape=cl_dist.shape)
    zl = zl.astype('float32')
    return zl


def pointbar_surface(cl_dist, z, W, D):
    """
    Create Riverbed point-bar surface.
    :param cl_dist: (numpy.2darray) - Distance from center-line on X-Y plane.
    :param z: (numpy.1darray) - z-coordinates of center-line.
    :param W: (Float) - River's width.
    :param D: (Float) - River's depth.
    :return: zpb: (numpy.2darray) - z-coordinates of point-bar surface.
    """
    if len(z[z - z[0] != 0]):
        raise ValueError('Can not process center-line with changing z-coordinates.')
    zpb = z[0] + D * np.exp(-(cl_dist ** 2) / (2 * (W / 4) ** 2))
    zpb = zpb.astype('float32')
    return zpb


def levee_surface(cl_dist, h_levee, w_levee, W, tp):
    """
    Create natural levee surface.
    :param cl_dist: (numpy.2darray) - Distance from center-line on X-Y plane.
    :param h_levee: (Float) - The Maximum thickness of levee.
    :param w_levee: (Float) - The width of levee.
    :param W: (Float) - River's width.
    :param tp: (numpy.2darray) - Topography.
    :return: zlv: (numpy.2darray) - z-coordinates of levee surface.
    """
    th1 = -2 * h_levee / w_levee * (cl_dist - W / 2 - w_levee / 2)
    th2 = np.ones(shape=cl_dist.shape) * h_levee
    th1, th2 = th1.astype('float32'), th2.astype('float32')
    th_levee = np.minimum(th1, th2)
    th_levee[th_levee < 0] = 0
    zlv = tp - th_levee
    return zlv


def kth_diag_indices(a, k):
    """
    Function for finding diagonal indices with k offset.
    [From https://stackoverflow.com/questions/10925671/numpy-k-th-diagonal-indices]
    :param a: (numpy.2darray) - The input array.
    :param k: (Integer) - The offset. For example, k=1 represents the diagonal elements 1 step below the main diagonal,
              k=-1 represents the diagonal elements 1 step above the main diagonal.
    :return: rows: (numpy.1darray) - The row indices.
             col: (numpy.1darray) - The column indices.
    """
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[:k], cols[-k:]
    elif k > 0:
        return rows[k:], cols[:-k]
    else:
        return rows, cols


def find_neck(x, y, delta_s, critical_dist, n_buffer=20):
    """
    Find the location of neck cutoff. Modified from Zoltan Sylvester's meanderpy.
    [https://github.com/zsylvester/meanderpy]
    :param x: (numpy.1darray) - x-coordinates of center-line.
    :param y: (numpy.1darray) - y-coordinates of center-line.
    :param critical_dist: (Float) - The critical distance. Cutoff occurs when distance of two points on center-line is
                          shorter than (or equal to) the critical distance.
    :param delta_s: (Float) - Distance between two consecutive points on center-line.
    :param n_buffer: (Integer) - Number of buffer points, preventing that cutoff occurs where there's no bend.
    :return: ind1: (numpy.1darray) - Indexes of center-line coordinates array where the cutoffs start.
             ind2: (numpy.1darray) - Indexes of center-line coordinates array where the cutoffs end.
    """
    # Number of neighbors that will be ignored for neck search.
    n_ignore = int((critical_dist + n_buffer * delta_s) / delta_s)
    # Compute Euclidean distance between each pair of points on center-line.
    dist = distance.cdist(np.array([x, y], dtype='float32').T, np.array([x, y], dtype='float32').T, metric='euclidean')
    # Set distances greater than critical distance to NAN.
    dist[dist > critical_dist] = np.NAN
    # Set ignored neighbors' distance to NAN.
    for i in range(-n_ignore, n_ignore + 1):
        rows, cols = kth_diag_indices(dist, i)
        dist[rows, cols] = np.NAN
    # Find where the distance is lower than critical distance.
    r, c = np.where(~np.isnan(dist))
    # Take only the points measured downstream.
    ind1 = r[np.where(r < c)[0]]
    ind2 = c[np.where(r < c)[0]]
    return ind1, ind2


def execute_cutoff(x, y, z, delta_s, critical_dist):
    """
    Execute cutoff on center-line. Modified from Zoltan Sylvester's meanderpy.
    [https://github.com/zsylvester/meanderpy]
    :param x: (numpy.1darray) - x-coordinates of center-line.
    :param y: (numpy.1darray) - y-coordinates of center-line.
    :param z: (numpy.1darray) - z-coordinates of center-line.
    :param delta_s: (Float) - Distance between two consecutive points on center-line.
    :param critical_dist: (Float) - The critical distance. Cutoff occurs when distance of two points on center-line is
                          shorter than (or equal to) the critical distance.
    :return: xc: (List) - x-coordinates of cutoffs.
             yc: (List) - y-coordinates of cutoffs.
             zc: (List) - z-coordinates of cutoffs.
             x: (numpy.1darray) - x-coordinates of center-line after cutoff.
             y: (numpy.1darray) - y-coordinates of center-line after cutoff.
             z: (numpy.1darray) - z-coordinates of center-line after cutoff.
    """
    xc = []
    yc = []
    zc = []
    ind1, ind2 = find_neck(x, y, delta_s, critical_dist)
    while len(ind1) > 0:
        xc.append(x[ind1[0]:ind2[0] + 1])  # x-coordinates of cutoffs.
        yc.append(y[ind1[0]:ind2[0] + 1])  # y-coordinates of cutoffs.
        zc.append(z[ind1[0]:ind2[0] + 1])  # z-coordinates of cutoffs.
        x = np.concatenate((x[:ind1[0] + 1], x[ind2[0]:]))  # x-coordinates of center-line after cutoff.
        y = np.concatenate((y[:ind1[0] + 1], y[ind2[0]:]))  # y-coordinates of center-line after cutoff.
        z = np.concatenate((z[:ind1[0] + 1], z[ind2[0]:]))  # z-coordinates of center-line after cutoff.
        ind1, ind2 = find_neck(x, y, delta_s, critical_dist)
    return xc, yc, zc, x, y, z


def plot_channel2D(channel_obj, oxbow_obj, title=None, interval=10):
    """
    Plot channel's migration on X-Y plane.
    :param channel_obj: (List) - The channel objects.
    :param oxbow_obj: (List) - The oxbow-lake objects.
    :param title: (String) - The title of the figure.
    :param interval: (Integer) - Plot channel for every "interval" channels.
    """
    # Set figure parameters.
    plt.figure(figsize=(16, 9))
    plt.xlabel('X(m)', fontsize=15)
    plt.ylabel('Y(m)', fontsize=15)
    plt.axis('equal')
    if title is None:
        title = 'Meandering River Migration'
    plt.title(title, fontsize=20)
    plt.tick_params(labelsize=15)
    for i in range(0, len(channel_obj), interval):
        x, y = channel_obj[i].x, channel_obj[i].y  # Get center-line coordinates.
        W = channel_obj[i].W  # Get river's width.
        xb, yb = channel_bank(x, y, W)  # Compute bank coordinates.
        # Make the banks a closed curve.
        xb = np.hstack((xb[:, 0], xb[:, 1][::-1]))
        yb = np.hstack((yb[:, 0], yb[:, 1][::-1]))
        if i == 0:
            plt.fill(xb, yb, facecolor='grey', edgecolor='black', alpha=1.0)
        elif i + interval >= len(channel_obj):
            plt.fill(xb, yb, facecolor='blue', edgecolor='black', alpha=1.0)
            # plt.plot(x, y, 'ko--', linewidth=2, markersize=10)
        else:
            plt.fill(xb, yb, facecolor='yellow', edgecolor='black', alpha=0.5)
    # If there are oxbow-lakes, plot oxbow-lakes.
    for i in range(len(oxbow_obj)):
        # Get oxbow-lake center-line coordinates.
        # Note that different oxbow-lake center-line coordinates are stored in different arrays.
        xo, yo = oxbow_obj[i].x, oxbow_obj[i].y  # xc: List[array0, array1, ...]
        W = oxbow_obj[i].W  # Get oxbow-lake width.
        if len(xo) > 0:
            n_oxbow = len(xo)  # Number of oxbow-lakes.
            for j in range(n_oxbow):
                xbo, ybo = channel_bank(xo[j], yo[j], W)  # Compute bank coordinates of oxbow-lakes.
                # Make the banks a closed curve.
                xbo = np.hstack((xbo[:, 0], xbo[:, 1][::-1]))
                ybo = np.hstack((ybo[:, 0], ybo[:, 1][::-1]))
                plt.fill(xbo, ybo, facecolor='deepskyblue', edgecolor='black', alpha=1.0)
    plt.show()
