import random

import numpy as np

from truthlens.seed import set_seed


def test_set_seed_makes_random_draws_repeatable():
    set_seed(42)
    first = (random.random(), np.random.rand())

    set_seed(42)
    second = (random.random(), np.random.rand())

    assert first == second