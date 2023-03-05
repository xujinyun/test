import numpy as np
import matplotlib.pyplot as plt
import math

from map_reader import MapReader
from sensor_model_my import SensorModel
from main import init_particles_freespace
from scipy import special
from scipy.stats import norm

def test_hyperparameter():
    map_obj = MapReader('../data/map/wean.dat')
    occupancy_map = map_obj.get_map()

    X = init_particles_freespace(1, occupancy_map)

    sensor = SensorModel(occupancy_map)
    X = X.reshape(1, -1)

    def mixture(z_t, zt_ast):
        def cdf(x, u, sigma):
            return 0.5 * (1 + special.erf((x - u) / (sigma * math.sqrt(2))))
        eta = cdf(sensor._max_range, zt_ast, sensor._sigma_hit) - cdf(0, zt_ast, sensor._sigma_hit)
        p_hit = norm.pdf(z_t, zt_ast, sensor._sigma_hit) / eta
        p_hit[z_t < 0] = 0
        p_hit[z_t > sensor._max_range] = 0

        p_short = sensor._lambda_short * np.exp(-sensor._lambda_short * z_t)
        anti_explosion = 1e-9
        p_short = p_short / (1 - np.exp(-sensor._lambda_short * zt_ast) + anti_explosion)
        p_short[z_t < 0] = 0
        p_short[z_t > zt_ast] = 0

        p_max = np.ones(z_t.shape)
        p_max[z_t < sensor._max_range] = 0

        p_rand = np.ones(z_t.shape) / sensor._max_range
        p_rand[z_t < 0] = 0
        p_rand[z_t > sensor._max_range] = 0

        return p_hit, p_short, p_max, p_rand

    

    z = np.arange(sensor._max_range + 2).astype(float)

    p_hit, p_short, p_max, p_rand = mixture(z, 500)
    
    p = sensor._z_hit * p_hit + sensor._z_short * p_short + sensor._z_max * p_max + sensor._z_rand * p_rand
    
    fig = plt.figure()
    plt.bar(np.arange(len(p)), p)
    plt.show()


if __name__ == '__main__':
    test_hyperparameter()