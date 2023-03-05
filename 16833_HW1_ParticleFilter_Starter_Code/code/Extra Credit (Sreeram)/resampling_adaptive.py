'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)
        M=X_bar.shape[0]
        weight=X_bar[:,-1]/np.sum(X_bar[:,-1])

        choice=np.random.choice(M,M,p=weight)
        X_bar_resampled=X_bar[choice,:]
        X_bar_resampled[:,-1]=1

        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)
        # M equals number of particles
        M=X_bar.shape[0]
        invM=1/M

        # Low variance resampling 
        # select random value from 0 to 1/M and divide all weights by M
        r=np.random.uniform(low=0,high=invM)
        weight = X_bar[:,-1]/np.sum(X_bar[:,-1])
        # Initial cumulative sum is first weight value
        c=weight[0]
        # i tracks the index of the current particle to sample
        i=0
        # variables to track the number of times a point has been sampled, the max number of times we want a point sampled, and the current row in the return matrix
        number_times_used = 0
        max_uses = 10
        row_current = 0
        # Iterate over the particles
        for m in range(M):
            # Calculate expected weight if everything was balanced plus random value
            # If cumulative weight sum is under expectation, iterate i to a higher weight particle, and reset number_times_used
            # If cumulative weight sum is higher than expected, the current particle can be copied to the current row
            # in the return matrix if the number of times it has been used is below the max
            u = r + m * invM
            while u > c:
                i += 1
                c += weight[i]
                number_times_used = 0
            else:
                 if number_times_used < max_uses:
                    X_bar_resampled[row_current,:]=X_bar[i,:]
                    row_current += 1
                    number_times_used += 1
        # Return the matrix up to current row, potentially smaller if some points have much higher weight than others
        X_bar_resampled[:row_current, -1] = 1
        return X_bar_resampled[:row_current, :]
