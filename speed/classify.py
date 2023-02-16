from cmath import nan
import os
from selectors import EpollSelector
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
    n_NAN = 0

    for i in range(11, len(gau)):
        if pd.isna(float(gau[2][i])) == True:
            n_NAN += 1
    if n_NAN >= 1:
        return NANS
    if n_data_inf >= 1:  # explosion > 1s
        return EXPLOSION
    else:
        return NORMAL

if __name__ == "__main__":

    # Read gaussian fitting result (gau.dat)
    filename = ''

    # Classify bump type
    if os.path.exists(filename):            
        data = pd.read_csv(filename, delimiter=' ', header=None)
        result = classify_bump(filename)
        if result == EXPLOSION or result == ROTATION_FAILED:
            print("Failed")