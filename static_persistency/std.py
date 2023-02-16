from cmath import nan
import os
from turtle import color
from unittest.result import STDERR_LINE
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


NORMAL = 0
NOBUMP = 1
EXPLOSION = 2
ROTATION_FAILED = 3
NANS = 4 

def classify(filename):
    gau = pd.read_csv(filename, delim_whitespace=True, header=None)

    n_data_0 = len(gau[gau[1] < 1])  # amplitude < 1: no bump
    n_data_inf = len(gau[gau[3] > 50])  # width > 50: explosion
    n_NAN = 0

    for i in range(5, 101):
        if pd.isna(float(gau[1][i])) == True:
            n_NAN += 1
    if n_data_0 >= 10:  # no bump > 1s
        return NOBUMP
    elif n_data_inf >= 10:  # explosion > 1s
        return EXPLOSION
    elif n_NAN >= 10:
        return NANS
    else:
        return NORMAL
        # return (NORMAL, rotation_speed)

def std(filename):
    gau = pd.read_csv(filename, delim_whitespace=True, header=None)
    std=np.std(gau[2])
    return std


if __name__ == "__main__":
    # Read gaussian fitting result (gau.dat)
    filename=''
    
    # Classify bump type
    result=classify(filename)
    
    # If bump exists, evaluate the position std
    if result==0:
        score=std(filename)
        print(score)
