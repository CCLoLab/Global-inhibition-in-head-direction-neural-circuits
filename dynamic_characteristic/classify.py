from cmath import nan
import os
from turtle import color
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


def classify_bump(filename):
    # the file is Gaussian fitting result

    gau = pd.read_csv(filename, delim_whitespace=True, header=None)

    n_data_0 = len(gau[gau[1] < 1])  # amplitude < 1: no bump
    n_data_inf = len(gau[gau[3] > 50])  # width > 50: explosion
    n_NAN1 = 0
    n_NAN2 = 0
    
    # 5s~5.5s, first visual input switch
    for i in range(50, 55):
        if pd.isna(float(gau[2][i])) == True:
            n_NAN1 += 1
    # 8s~8.5s, second visual input switch
    for i in range(80, 85):
        if pd.isna(float(gau[2][i])) == True:
            n_NAN2 += 1
    if n_data_inf >= 10:  # explosion > 1s
        return EXPLOSION
    if n_data_0 >= 10:  # explosion > 1s
        return NOBUMP
    elif n_NAN1 >= 1 or n_NAN2 >= 1:
        return NANS
    else:
        return NORMAL


if __name__ == "__main__":

    # Read gaussian fitting result (gau.dat)
    filename = ''

    if os.path.exists(filename):
        result = classify_bump(filename)
        if result == NANS:
            print("jump")