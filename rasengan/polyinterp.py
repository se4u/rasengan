#!/usr/bin/env python
'''
| Filename    : polyinterp.py
| Description : Polynomial interpolation of data points.
| Author      : Pushpendre Rastogi
| Created     : Tue Sep  6 00:10:47 2016 (-0400)
| Last-Updated: Tue Sep  6 00:11:35 2016 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 1
Copied from http://hsakamoto.com/docs/numerical/ch3/301.html
September 6 2016
'''
import numpy as np
import matplotlib.pyplot as plt


def PI(x_data, y_data):
    """ return the polynomially interpolated function
    for a given set of data points by Newton's method
    """
    a = coeff(x_data, y_data)
    n = len(x_data)

    def f(xi):
        """ polynomially interpolated function
        """
        v = a[n - 1]  # initial value
        for k in range(1, n):
            # compute backwardly
            v = a[-k - 1] + (xi - x_data[-k - 1]) * v
        return v
    return f


def coeff(x_data, y_data):
    """ return the coefficients for Newton's polynomial
    """
    x = np.array(x_data)
    y = np.array(y_data)
    m = len(y)
    a = y.copy()  # initial value (a[0] = y[0])
    for k in range(m - 1):
        for i in range(k, m - 1):
            # compute divided difference of k-th order for all i >= k
            a[i + 1] = (a[i + 1] - a[k]) / (x[i + 1] - x[k])
    return a  # coefficient
