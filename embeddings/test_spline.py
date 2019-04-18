import os
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.interpolate as si
from mpl_toolkits.mplot3d import Axes3D


def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """

    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count + degree + 1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree, 1, degree)

    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree, 1, count - 1)

    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0 - degree, count + degree + degree - 1)
    else:
        kv = np.clip(np.arange(count + degree + 1) - degree, 0, count - degree)

    # Calculate query range
    u = np.linspace(periodic, (count - degree), n)

    # Calculate result
    return np.array(si.splev(u, (kv, cv.T, degree))).T


if __name__ == '__main__':

    tract_name = sys.argv[0]
    if not os.path.exists(tract_name):
        sys.exit('Wrong tract path')

    tract = nib.streamlines.load('tract_name')
    s = tract.streamlines[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

    cv = s

    ax.plot(cv[:, 0], cv[:, 1], cv[:, 2], 'o-', label='Control Points')

    for d in range(1, 4):
        p = bspline(cv, n=30, degree=d, periodic=False)
        x, y, z = p.T
        ax.plot(
            x, y, z, 'k-', label='Degree %s' % d, color=colors[d % len(colors)])

    plt.minorticks_on()
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()
