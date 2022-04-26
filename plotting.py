import gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from collections import defaultdict

"""
ref: https://github.com/dennybritz/reinforcement-learning/blob/master/lib/plotting.py
"""

def plot_values(V, title="Value Function"):

    def get_Z(x, y, usable_ace):
        if (x,y,usable_ace) in V:
            return V[x,y,usable_ace]
        else:
            return 0

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)

        Z = np.array([get_Z(x,y,usable_ace) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Current Sum')
        ax.set_ylabel('Dealer Showing Card')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title("{} Usable Ace".format(title))
    get_figure(True, ax)
    ax = fig.add_subplot(212, projection='3d')
    ax.set_title("{} No usable Ace".format(title))
    get_figure(False, ax)
    plt.savefig("{}.png".format(title))
    plt.show()
