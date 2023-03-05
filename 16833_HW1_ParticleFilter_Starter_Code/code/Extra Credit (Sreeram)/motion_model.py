'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.0001
        self._alpha2 = 0.0001
        self._alpha3 = 0.013
        self._alpha4 = 0.005


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        # Algorithm sample_motion_model_odometry on pg 110, Chapter 5.4.1 of Probabilistic Robotics by Thrun, Burgard, and Fox.
        
        num = len(x_t0)

        delta_rot_1 = np.arctan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        delta_trans = np.sqrt(np.power(u_t0[1] - u_t1[1], 2) + np.power(u_t0[0] - u_t1[0], 2))
        delta_rot_2 = u_t1[2] - u_t0[ 2] - delta_rot_1

        hat_dr1 = delta_rot_1 - np.random.normal(0, np.sqrt(self._alpha1 * np.power(delta_rot_1, 2) + self._alpha2 * np.power(delta_trans, 2)), num)

        hat_dt = delta_trans - np.random.normal(0, np.sqrt(self._alpha3 * np.power(delta_trans, 2) +
                                                                   self._alpha4 * np.power(delta_rot_1, 2) + 
                                                                   self._alpha4 * np.power(delta_rot_2, 2)), num)
                                                                   
        hat_dr2 = delta_rot_2 - np.random.normal(0, np.sqrt(self._alpha1 * np.power(delta_rot_2, 2) + self._alpha2 * np.power(delta_trans, 2)), num)

        xp = x_t0[:, 0] + np.multiply(hat_dt, np.cos(x_t0[:, 2] + hat_dr1))
        yp = x_t0[:, 1] + np.multiply(hat_dt, np.sin(x_t0[:, 2] + hat_dr1))
        tp = x_t0[:, 2] + hat_dr1 + hat_dr2

        return np.transpose(np.vstack((xp.flatten(), yp.flatten(), tp.flatten())))

        # return np.random.rand(3)
