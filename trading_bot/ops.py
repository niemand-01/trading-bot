import os
import math
import logging

import numpy as np


def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)


def get_state(data, t, n_days, position_state=0.0):
    """Returns an n-day state representation ending at time t with position state
    
    Args:
        data: Stock price data
        t: Current time index
        n_days: Number of days for state representation
        position_state: Position state (-1.0 for short, 0.0 for flat, 1.0 for long)
    """
    d = t - n_days + 1
    block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[0: t + 1]  # pad with t0
    res = []
    for i in range(n_days - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    # Add position state as the last feature
    res.append(position_state)
    return np.array([res])
