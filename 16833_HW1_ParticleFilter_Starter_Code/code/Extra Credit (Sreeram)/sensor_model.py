'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader
from scipy.special import erf


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 1.1
        self._z_short = 0.1
        # Detection failure
        self._z_max = 0.0003
        # Random mesearument
        self._z_rand = 1

        # Intrinsic noise parameter for local mesearument noise
        self._sigma_hit = 100
        # Intrinsic parameter for unexpected objects
        self._lambda_short = 0.01

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 10
        self._K = 180 // self._subsampling      # Number of laser beams
        self._rate = 1                          # ray casting rate

        # Get occupancy_map
        self._map = occupancy_map

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """

        # Step 1: Transformation between sensor and odometry
        # print("state shape:", x_t1.shape)
        x_t_sensor = x_t1.copy()
        # print("sensor shape:", x_t_sensor.shape)
        # transformation to world frame
        x_t_sensor[:, 0] += 25*np.cos(x_t_sensor[:, 2])
        x_t_sensor[:, 1] += 25*np.sin(x_t_sensor[:, 2])
        x_t_sensor[:, :2] /= 10     # Unit comfirm
        x_t_sensor[:, 2] -= np.pi / 2
        
        # Step 2: ray casting Get beam measurements zt_star 
        # denote the “true” range of the object measured by z_t
        # Creat a laser measurement with size K x m X 3, 
        # m is the number of particles, K is samples from angle 0 to angle 180
        ray = np.repeat(x_t_sensor[None, ...], self._K, 0)
        ray[..., 2] += np.arange(0, 180, self._subsampling).reshape(-1, 1) * np.pi / 180
        ray[..., 2] = ray[..., 2] % ( 2 * np.pi)
        ray = ray.reshape((-1, 3))    # shape (Kxm)*3
        # print(ray.shape)
        # Step 2.1: Iterate throuh each laser ray can check collision
        zt_star = np.zeros(ray.shape[0])
        beam_active = np.ones(ray.shape[0]).astype(bool) 
        # print(beam_active.shape)
        while np.sum(beam_active):                 # Still have live beams, keep expanding
            # Keep expanding
            ray[beam_active, 0] += self._rate * np.cos(ray[beam_active, 2])
            ray[beam_active, 1] += self._rate * np.sin(ray[beam_active, 2])        
            zt_star[beam_active] += self._rate

            # Inbounded check
            active_index = np.where(beam_active)[0]
            x = np.round(ray[beam_active, 0]).astype(int)
            y = np.round(ray[beam_active, 1]).astype(int)
            inbound = (x >= 0) & (x < self._map.shape[1]) & (y >= 0) & (y < self._map.shape[0])
            beam_active[active_index[np.where(~inbound)[0]]] = False

            # Collision check
            active_index = active_index[inbound]
            ind = np.where(inbound)[0] # only check inbounded point
            x_inbound, y_inbound = x[ind], y[ind] # position of inbounded point
            occupied = ((self._map[y_inbound, x_inbound] > self._min_probability) 
                    |   (self._map[y_inbound, x_inbound] == -1))
            beam_active[active_index[np.where(occupied)[0]]] = False

            # Range check
            out_range = np.where(zt_star * 10 > self._max_range)[0]
            beam_active[out_range] = False
            zt_star[out_range] = self._max_range / 10


        zt_star = 10 * zt_star.reshape((self._K, -1)).T
        z_t = z_t1_arr[0:181:self._subsampling].copy()
        z_t = np.repeat(z_t[None, ...], len(x_t1), 0)
        
        # Step 3: Merge the density
        # loop through 1-K
            # p = sum(z*p) for hit, short, max, and rand
            # prob_zt1 = prob_zt1*p
        eta_hit1 = self.Gaussian_cdf(self._max_range, zt_star, self._sigma_hit)
        eta_hit2 = self.Gaussian_cdf(0, zt_star, self._sigma_hit)

        eta_hit = self.Gaussian_cdf(self._max_range, zt_star, self._sigma_hit) - self.Gaussian_cdf(0, zt_star, self._sigma_hit)
        p_hit = norm.pdf(z_t, zt_star, self._sigma_hit)
        p_hit /= eta_hit
        p_hit[np.logical_or(z_t < 0, z_t > self._max_range)] = 0

        p_short = self._lambda_short * np.exp(-self._lambda_short * z_t)
        eta_short = 1 - np.exp(-self._lambda_short * zt_star) + 1e-8
        p_short /= eta_short
        p_short[np.logical_or(z_t < 0, z_t > zt_star)] = 0

        p_max = np.ones(z_t.shape)
        p_max[z_t < self._max_range] = 0

        p_rand = np.ones(z_t.shape) / self._max_range
        p_rand[np.logical_or(z_t < 0, z_t > self._max_range)] = 0

        prob_zt1 = self._z_hit * p_hit + self._z_short * p_short + self._z_max * p_max + self._z_rand * p_rand 
        # convert probability to log for muliplication
        prob_zt1 = np.sum(np.log(prob_zt1), 1)
        prob_zt1 -= np.max(prob_zt1)
        prob_zt1 = np.exp(prob_zt1)
        prob_zt1 /= np.sum(prob_zt1)

        return prob_zt1

    def Gaussian_cdf(self, x, u, sigma):
        return (1 + erf((x - u)/(sigma*math.sqrt(2))) ) / 2