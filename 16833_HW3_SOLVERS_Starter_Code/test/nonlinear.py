'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import argparse
import matplotlib.pyplot as plt
from solvers import *
from utils import *


def warp2pi(angle_rad):
    """
    Warps an angle in [-pi, pi]. Used in the update step.
    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad = angle_rad - 2 * np.pi * np.floor(
        (angle_rad + np.pi) / (2 * np.pi))
    return angle_rad


def init_states(odoms, observations, n_poses, n_landmarks):
    '''
    Initialize the state vector given odometry and observations.
    '''
    traj = np.zeros((n_poses, 2))
    landmarks = np.zeros((n_landmarks, 2))
    landmarks_mask = np.zeros((n_landmarks), dtype=bool)

    for i in range(len(odoms)):
        traj[i + 1, :] = traj[i, :] + odoms[i, :]

    for i in range(len(observations)):
        pose_idx = int(observations[i, 0])
        landmark_idx = int(observations[i, 1])

        if not landmarks_mask[landmark_idx]:
            landmarks_mask[landmark_idx] = True

            pose = traj[pose_idx, :]
            theta, d = observations[i, 2:]

            landmarks[landmark_idx, 0] = pose[0] + d * np.cos(theta)
            landmarks[landmark_idx, 1] = pose[1] + d * np.sin(theta)

    return traj, landmarks


def odometry_estimation(x, i):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from (odometry between pose i and i+1)
    \return odom Odometry (\Delta x, \Delta) in the shape (2, )
    '''
    # TODO: return odometry estimation
    # odom = np.zeros((2, ))

    odom = x[2*i+2 : 2*i+4] - x[2*i : 2*i+2]
    print(odom.shape)

    return odom


def bearing_range_estimation(x, i, j, n_poses):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return obs Observation from pose i to landmark j (theta, d) in the shape (2, )
    '''
    # TODO: return bearing range estimations
    # obs = np.zeros((2, ))

    rx,ry = x[2*i : 2*i+2]
    lx,ly = x[2*n_poses+2*j : 2*n_poses+2*j+2]
    obs = [np.arctan2(ly-ry,lx-rx), np.sqrt((lx-rx)**2+(ly-ry)**2)]

    return np.array(obs)


def compute_meas_obs_jacobian(x, i, j, n_poses):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return jacobian Derived Jacobian matrix in the shape (2, 4)
    '''
    # TODO: return jacobian matrix
    jacobian = np.zeros((2, 4))
    rx,ry = x[2*i : 2*i+2]
    lx,ly = x[2*n_poses+2*j : 2*n_poses+2*j+2]
    dx = lx - rx
    dy = ly - ry
    jacobian[0] = np.array([ dy,-dx,-dy,dx])/(dx**2+dy**2)
    jacobian[1] = np.array([-dx,-dy,dx,dy]) / np.sqrt(dx**2+dy**2)


    return jacobian


def create_linear_system(x, odoms, observations, sigma_odom, sigma_observation,
                         n_poses, n_landmarks):
    '''
    \param x State vector x at which we linearize the system.
    \param odoms Odometry measurements between i and i+1 in the global coordinate system. Shape: (n_odom, 2).
    \param observations Landmark measurements between pose i and landmark j in the global coordinate system. Shape: (n_obs, 4).
    \param sigma_odom Shared covariance matrix of odometry measurements. Shape: (2, 2).
    \param sigma_observation Shared covariance matrix of landmark measurements. Shape: (2, 2).

    \return A (M, N) Jacobian matrix.
    \return b (M, ) Residual vector.
    where M = (n_odom + 1) * 2 + n_obs * 2, total rows of measurements.
          N = n_poses * 2 + n_landmarks * 2, length of the state vector.
    '''

    n_odom = len(odoms)
    n_obs = len(observations)

    M = (n_odom + 1) * 2 + n_obs * 2
    N = n_poses * 2 + n_landmarks * 2

    A = np.zeros((M, N))
    b = np.zeros((M, ))

    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))

    # Prepare Jacobians
    Ho = np.array([[-1,0,1,0],[0,-1,0,1]]) #odometry measurement

    # TODO: First fill in the prior to anchor the 1st pose at (0, 0)
    A[:2,:2] = np.eye(2)

    # TODO: Then fill in odometry measurements
    for i in range(n_odom):
        A[2+2*i:2+2*i+2, 2*i:2*i+4] = sqrt_inv_odom @ Ho
        print('x:',x.shape)
        # print(i, odometry_estimation(x,i).shape)
        d = odoms[i] - odometry_estimation(x,i)
        # print("d: ", d.shape)
        # print("inv odom: ", sqrt_inv_odom.shape) 
        # print("b: ", b[2+2*i:2+2*i+2].shape)
        b[2+2*i:2+2*i+2] = (sqrt_inv_odom @ d[:,None]).squeeze()


    # TODO: Then fill in landmark measurements
    idx_rowobs = (n_odom +1)*2
    for idx in range(n_obs):
        i,j = observations[idx,:2].astype(int) #pose i and landmark j
        Hl = compute_meas_obs_jacobian(x,i,j,n_poses)

        A[idx_rowobs+2*idx : idx_rowobs+2*idx+2, 2*i:2*i+2] = sqrt_inv_obs @ Hl[:,:2]
        A[idx_rowobs+2*idx : idx_rowobs+2*idx+2, 2*(n_poses+j):2*(n_poses+j)+2] = sqrt_inv_obs @ Hl[:,2:]
        
        d = observations[idx, 2:] - bearing_range_estimation(x,i,j,n_poses)
        d[0] = warp2pi(d[0])
        b[idx_rowobs+2*idx : idx_rowobs+2*idx+2] = (sqrt_inv_obs @ d[:,None]).squeeze()

    return csr_matrix(A), b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../data/2d_nonlinear.npz')
    parser.add_argument(
        '--method',
        nargs='+',
        choices=['default', 'pinv', 'qr', 'lu', 'qr_colamd', 'lu_colamd'],
        default=['default'],
        help='method')

    args = parser.parse_args()

    data = np.load(args.data)

    # Plot gt trajectory and landmarks for a sanity check.
    gt_traj = data['gt_traj']
    gt_landmarks = data['gt_landmarks']
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-')
    plt.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c='b', marker='+')
    plt.show()

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odom = data['odom']
    observations = data['observations']
    sigma_odom = data['sigma_odom']
    sigma_landmark = data['sigma_landmark']

    # Initialize: non-linear optimization requires a good init.
    for method in args.method:
        print(f'Applying {method}')
        traj, landmarks = init_states(odom, observations, n_poses, n_landmarks)
        print('Before optimization')
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)

        # Iterative optimization
        x = vectorize_state(traj, landmarks)
        for i in range(10):
            A, b = create_linear_system(x, odom, observations, sigma_odom,
                                        sigma_landmark, n_poses, n_landmarks)
            dx, _ = solve(A, b, method)
            x = x + dx.squeeze() # pay attention to this line

        traj, landmarks = devectorize_state(x, n_poses)
        print('After optimization')
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)
