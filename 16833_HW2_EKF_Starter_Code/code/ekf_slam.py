'''
    Initially written by Ming Hsiao in MATLAB
    Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import re
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)


def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array(
            [3 * np.sqrt(a) * np.cos(phi[i]),
             3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    # plt.waitforbuttonpress(0)


def draw_traj_and_map(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + k * 2:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'r')
    else:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + 2 * k:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'g')

    plt.draw()
    # plt.waitforbuttonpress(0)


def warp2pi(angle_rad):
    """
    TODO: warps an angle in [-pi, pi]. Used in the update step.

    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad = angle_rad % (2 * np.pi)  # restrict angle to [0, 2pi)
    if (angle_rad > np.pi):
        angle_rad -= 2 * np.pi
    return angle_rad


def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    TODO: initialize landmarks given the initial poses and measurements with their covariances
    \param init_measure Initial measurements in the form of (beta0, l0, beta1, l1, ...).
    \param init_measure_cov Initial covariance matrix of shape (2, 2) per landmark given parameters.
    \param init_pose Initial pose vector of shape (3, 1).
    \param init_pose_cov Initial pose covariance of shape (3, 3) given parameters.

    \return k Number of landmarks.
    \return landmarks Numpy array of shape (2k, 1) for the state.
    \return landmarks_cov Numpy array of shape (2k, 2k) for the uncertainty.
    '''
    # print(init_measure)
    k = init_measure.shape[0] // 2

    landmark = np.zeros((2 * k, 1))
    landmark_cov = np.zeros((2 * k, 2 * k))
    
    for i in range(k):
        beta = init_measure[2*i]
        r = init_measure[2*i + 1]
        # Get landmark state in world frame
        landmark[2*i] = init_pose[0] + r * np.cos(init_pose[2] + beta)
        landmark[2*i+1] = init_pose[1] + r * np.sin(init_pose[2] + beta)
        theta = init_pose[2][0]
        r = r[0]
        beta = beta[0]

        # Update covraince
        # H = np.array([[-r * np.sin(init_pose[2] + beta), np.cos(init_pose[2] + beta)],
        #                [r * np.cos(init_pose[2] + beta), np.sin(init_pose[2] + beta)]])
        # H = H[:, :, 0]
        # print(init_pose)
        # print(init_pose[2])
        # print(theta)
        # print(beta)
        # print(r)
        # print(-r * np.sin(theta + beta))
        J = np.array([[1, 0, -r * np.sin(theta + beta), -r * np.sin(theta + beta), np.cos(theta + beta)],
                      [0, 1, r * np.cos(theta + beta), r * np.cos(theta + beta), np.sin(theta + beta)]])
        
        # landmark_cov[2*i : 2*i+2, 2*i : 2*i+2] = H @ init_measure_cov @ H.T
        landmark_cov[2*i : 2*i+2, 2*i : 2*i+2] = J @ \
            (np.block([[init_pose_cov, np.zeros((3, 2 ))], [np.zeros((2, 3)), init_measure_cov]]) )\
            @ J.T

    return k, landmark, landmark_cov


def predict(X, P, control, control_cov, k):
    '''
    TODO: predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance of shape (3, 3) in the (x, y, theta) space given the parameters.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    '''
    
    # p_t = X[: 3]    # x_t, y_t, theta_t
    x, y, theta = X[0], X[1], X[2]
    theta = theta[0]
    d_t, alpha_t = control
    d_t = d_t[0]
    alpha_t  = alpha_t[0]

    # Prediction of pose at time t+1 only with control input
    X_input = np.zeros((3 + 2*k, 1))
    X_input[0] = d_t * np.cos(theta)
    X_input[1] = d_t * np.sin(theta)
    X_input[2] = alpha_t
    X_pre = X + X_input

    d_t, alpha_t = control
    d_t = d_t[0]
    alpha_t  = alpha_t[0]


    B = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0], 
                   [0, 0, 1]])
    R = np.zeros((3 + 2*k, 3 + 2*k))
    R[0:3 , 0:3] = B @ control_cov @ B.T

    G = np.eye(3 + 2*k)
    G[0:3, 0:3] = np.array([[1, 0, -d_t * np.sin(theta)],
                   [0, 1, d_t * np.cos(theta)],
                   [0, 0, 1]])
    
    P_pre = G @ P @ G.T + R

    
    return X_pre, P_pre


def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    TODO: update step in EKF SLAM with derived Jacobians.
    \param X_pre Predicted state vector of shape (3 + 2k, 1) from the predict step.
    \param P_pre Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    \param measure Measurement signal of shape (2k, 1).
    \param measure_cov Measurement covariance of shape (2, 2) per landmark given the parameters.
    \param k Number of landmarks.

    \return X Updated X state of shape (3 + 2k, 1).
    \return P Updated P covariance of shape (3 + 2k, 3 + 2k).
    '''
    X = np.copy(X_pre)
    P = np.copy(P_pre)
    Pm = np.zeros((2*k, 1))          # Predict the measurement of bearing and range
    H = np.zeros((2*k, 3+2*k))       # Measurement Jacobian
    R = np.zeros((2*k, 2*k))
    # print("Input: ", X_pre)
    for i in range(k):
        dx = X_pre[3 + 2*i] - X_pre[0]          # lx - x
        dy = X_pre[3 + 2*i + 1] - X_pre[1]      # ly - y
        dx = dx[0]
        dy = dy[0]
        eta = dx**2 + dy**2
        eta_sqrt = np.sqrt(eta)

        # Predict measurement for landmark
        Pm[i*2] = warp2pi(np.arctan2(dy, dx) - X_pre[2])
        Pm[i*2+1] = eta_sqrt

        # Add to measurement covariance
        R[2*i:2*i+2, 2*i:2*i+2] = measure_cov

        # Jacobian to robot pose
        Hp = np.array([[dy/eta, -dx/eta, -1],
                       [-dx/eta_sqrt, -dy/eta_sqrt, 0]])
        H[i*2 : i*2+2, 0:3] = Hp

        # Jacobian to landmark
        Hl = np.array([[-dy/eta, dx/eta],
                       [dx/eta_sqrt, dy/eta_sqrt]])
        H[i*2 : i*2+2, 3+i*2 : 3+i*2+2] = Hl[:, :]
    
    # Kalman Gain
    K = P_pre @ H.T @ np.linalg.inv(H @ P_pre @ H.T + R)
    # print("measure: ", measure)
    # print("pose: ", Pm)
    # print("Gain: ", K)
    X = X + K @ (measure - Pm)
    P = (np.eye(3+2*k) - K @ H) @ P_pre

    return X, P


def evaluate(X, P, k):
    '''
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    '''
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)
    plt.scatter(l_true[0::2], l_true[1::2])
    plt.draw()
    plt.waitforbuttonpress(0)

    # Compute Euclidean distance
    x_true = l_true.reshape(-1, 2)
    x_est = X[3:].reshape(-1, 2)
    euclidean_dist = np.linalg.norm(x_true - x_est, axis=1)
    for i in range(k):
        print(f'Euclidean distance for landmark {i+1}: {euclidean_dist[i]}')

    # Compute Mahalanobis distance for each landmark
    mahalanobis_dist = np.zeros(k)
    # mahalanobis_dist_test = np.zeros(k)
    for i in range(k):
        P_landmark = P[3 + 2*i:3 + 2*i+2, 3 + 2*i:3 + 2*i+2]
        mahalanobis_dist[i] = np.sqrt((x_true[i] - x_est[i]).T @ np.linalg.inv(P_landmark) @ (x_true[i] - x_est[i]))

        print(f'Mahalanobis distance for landmark {i+1}: {mahalanobis_dist[i]}')


def main():
    # TEST: Setup uncertainty parameters
    # sig_x = 0.25
    # sig_y = 0.1
    # sig_alpha = 0.1
    # sig_beta = 0.01
    # sig_r = 0.08

    sig_x = 0.25
    sig_y = 0.1
    sig_alpha = 0.1
    sig_beta = 0.01
    sig_r = 0.08


    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file and read the initial measurements
    data_file = open("../data/data.txt")
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = np.expand_dims(arr, axis=1)
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    pose = np.zeros((3, 1))
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    ##########
    # TODO: initialize landmarks
    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                               pose_cov)

    # Setup state vector X by stacking pose and landmark states
    # Setup covariance matrix P by expanding pose and landmark covariances
    X = np.vstack((pose, landmark))
    P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                  [np.zeros((2 * k, 3)), landmark_cov]])

    # Plot initial state and covariance
    last_X = X
    draw_traj_and_map(X, last_X, P, 0)

    # Core loop: sequentially process controls and measurements
    for line in data_file:
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])

        # Control
        if arr.shape[0] == 2:
            print(f'{t}: Predict step')
            d, alpha = arr[0], arr[1]
            control = np.array([[d], [alpha]])

            ##########
            # TODO: predict step in EKF SLAM
            X_pre, P_pre = predict(X, P, control, control_cov, k)

            draw_traj_and_pred(X_pre, P_pre)

        # Measurement
        else:
            print(f'{t}: Update step')
            measure = np.expand_dims(arr, axis=1)

            ##########
            # TODO: update step in EKF SLAM
            X, P = update(X_pre, P_pre, measure, measure_cov, k)

            draw_traj_and_map(X, last_X, P, t)
            last_X = X
            t += 1
            # print(P)

    # EVAL: Plot ground truth landmarks and analyze distances
    evaluate(X, P, k)


if __name__ == "__main__":
    main()
