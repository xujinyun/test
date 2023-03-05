'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
from sklearn import cluster
import time


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    # print(output_path)
    plt.pause(0.00001)
    scat.remove()


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    TODO : Add your code here
    This version converges faster than init_particles_random
    """
    y, x = np.where(occupancy_map == 0)  # 0 means unoccupied
    ind = np.random.choice(len(x), num_particles, replace=False)

    # map is 800x800 with a resolution of 10 (i.e. real map is 8000x8000)
    y0_vals, x0_vals = y[ind] * 10.0, x[ind] * 10.0  # scale to the unit of the real map, cm
    theta0_vals = np.random.uniform(-np.pi, np.pi, num_particles)

    # initialize weights for all particles
    w0_vals = np.ones(num_particles, dtype=np.float64)
    w0_vals = w0_vals / num_particles


    X_bar_init = np.vstack((x0_vals, y0_vals, theta0_vals, w0_vals)).T
    
    return X_bar_init


if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='../../results')
    parser.add_argument('--num_particles', default=5000, type=int)
    # parser.add_argument('--num_particles', default=7, type=int)
    # parser.add_argument('--visualize', action="store_false")
    parser.add_argument('--visualize', default = True)
    parser.add_argument('--adaptive', default=False)
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    np.random.seed(249)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    # X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        
        visualize_map(occupancy_map)

    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue
        
        # ignore pure odometry measurements for (faster debugging)
        if ((time_stamp <= 0.0) | (meas_type == "O")):
            continue

        u_t1 = odometry_robot

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        # for m in range(0, num_particles):
        #     """
        #     MOTION MODEL
        #     """
        #     x_t0 = X_bar[m, 0:3]
        #     x_t1 = motion_model.update(u_t0, u_t1, x_t0)

        #     """
        #     SENSOR MODEL
        #     """
        #     if (meas_type == "L"):
        #         z_t = ranges
        #         w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
        #         X_bar_new[m, :] = np.hstack((x_t1, w_t))
            # else:
            #     X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        # Vectorized version
        """
        MOTION MODEL
        """
        x_t = motion_model.update(u_t0, u_t1, X_bar[:, :3])  # motion model

        """
        SENSOR MODEL
        """
        if meas_type == 'L':
            w_t = sensor_model.beam_range_finder_model(ranges, x_t)  # sensor model
            X_bar_new = np.hstack((x_t, w_t.reshape(-1, 1)))
            # exit()
        else:
            X_bar_new = np.hstack((x_t, X_bar[:, 3].reshape(-1, 1)))

        u_t0 = u_t1

        """
        RESAMPLING
        """
        if meas_type == 'L':
            X_bar = resampler.low_variance_sampler(X_bar_new)

        """
        Adaptive Number of Particles
        Reference: https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
        """
        if args.adaptive:
            X_bar_reduced = []
            db = cluster.DBSCAN(eps=15, min_samples=2).fit(X_bar[:, :2])
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            unique_labels = set(labels)
            for k in unique_labels:
                if k == -1:
                    continue  # skip noise
                class_member_mask = labels == k
                X_class_core_members = X_bar[class_member_mask & core_samples_mask]
                probs = X_class_core_members[:, -1]
                probs /= np.sum(probs)  # renormalize the weights of the selected particles in this cluster
                size = len(X_class_core_members)
                X_bar_reduced.append(X_class_core_members[np.random.choice(size, min(size, 400), replace=False, p=probs)])
            X_bar = np.concatenate(X_bar_reduced, 0)

        if args.visualize:
            
            visualize_timestep(X_bar, time_idx, args.output)
