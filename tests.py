import numpy as np
from numpy import random
import epic1d as ep

def test_cal_density():
    npart = 5000
    L = 100
    pos = random.uniform(0., L, npart)
    ncells = 20
    result_default = ep.calc_density(pos,ncells,L)
    vec_result = ep.calc_density_vec(pos,ncells,L)
    assert((result_default==vec_result).all)

if __name__ == "__main__":
    test_cal_density()
    print("Everything passed")
