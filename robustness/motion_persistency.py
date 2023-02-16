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
    n_NAN = 0

    for i in range(151, 221):
        if pd.isna(float(gau[1][i])) == True:
            n_NAN += 1
    if n_data_0 >= 10:  # no bump > 1s
        return NOBUMP
    elif n_data_inf >= 10:  # explosion > 1s
        return EXPLOSION
    elif n_NAN >= 5:
        return NANS
    else:
        rotation_speed = avg_rotate_speed(gau)
        if rotation_speed == -1:
            return ROTATION_FAILED
        else:
            return NORMAL
            # return (NORMAL, rotation_speed)


def avg_rotate_speed(gau):

    # NOTE: if the protocol rotates for long duration,
    #       we can calculate the average of rotation speed & reject outliers,
    #       but currently protocol only have one successful rotation segment (10-15s)

    # def reject_outliers(data, m=2):
    #     return data[abs(data - np.mean(data)) < m * np.std(data)]

    speed_arr = []
    has_start = False

    for i in range(101, 151):  # based on the protocol: 10s-15s is rotation in one direction
        if has_start is False and 11 < gau.iat[i, 2] < 15:  # detect L8 -> R8 direction
            has_start = True
            time_a = i

        if has_start is True and 0 < gau.iat[i, 2] < 4:
            has_start = False
            time_b = i
            
            dx = (gau.iat[time_b, 2] - gau.iat[time_a, 2])
            dt = (gau.iat[time_b, 0] - gau.iat[time_a, 0]) * 0.01
            speed_arr.append(dx / dt)  # region/s
        


    if not speed_arr:
        return -1  # no successful rotation
    else:
        speed_arr = np.abs(np.array(speed_arr))
        # speed_arr = reject_outliers(speed_arr, 3)

        speed_mean = np.mean(speed_arr)
        speed_mean_degree = speed_mean * 22.5

        return speed_mean_degree
        # return (np.mean(speed_arr) * 22.5, np.std(speed_arr) * 22.5)

def correct_rotation(locations: np.ndarray):
    start_idx = 0
    for i in range(len(locations)):
        if not pd.isna(locations[i]):
            start_idx = i
            break

    current = locations[start_idx]
    previous_idx = start_idx
    locations_corrected = [current]

    for i in range(start_idx+1, len(locations)):
        if pd.isna(locations[i]):
            locations_corrected.append(np.nan)
            continue

        diff = locations[i] - locations[previous_idx]
        previous_idx = i

        if diff > 8:
            diff -= 16
        elif diff < -8:
            diff += 16

        current += diff
        locations_corrected.append(current)

    locations_corrected = pd.Series(locations_corrected)
    locations_corrected = locations_corrected.interpolate()
    return locations_corrected.values


def linear_fitting(arr):
    x = range(len(arr))
    y = arr
    model = LinearRegression()
    model.fit(np.c_[x], np.c_[y])
    r_square = model.score(np.c_[x], np.c_[y])

    return r_square  # (r_square, eq)


def evaluate_bump_rotation(gau: pd.DataFrame):
    segment_1st = gau.iloc[100:150, 2].values  # 10~15s
    segment_2nd = gau.iloc[150:200, 2].values  # 15~20s

    segment_1st_corrected = correct_rotation(segment_1st)
    segment_2nd_corrected = correct_rotation(segment_2nd)

    r_square_1st = linear_fitting(segment_1st_corrected)
    r_square_2nd = linear_fitting(segment_2nd_corrected)

    return r_square_1st + r_square_2nd

if __name__ == "__main__":
    # gaussian fitting result file (gau.dat)
    filename = ''

    # Classify bump type
    result = classify_bump(filename)
    
    # If bump exists, evaluate the motion persistency score (r^2)
    if result == 0:
        gau = pd.read_csv(filename, delim_whitespace=True, header=None)
        score = evaluate_bump_rotation(gau)