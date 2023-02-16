import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import sys
from scipy.stats import linregress


mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["axes.titlesize"] = 24
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16
mpl.rcParams["xtick.major.size"] = 4
mpl.rcParams["ytick.major.size"] = 4
mpl.rcParams["xtick.major.width"] = 2
mpl.rcParams["ytick.major.width"] = 2
mpl.rcParams["font.family"] = "Arial"


def gau_fit(fr: pd.DataFrame, step=10):

    def is_continuous(_list):
        i = 0  # marker for different states

        for val in _list:
            if i == 0 and val != 0:
                i += 1

            elif i == 1 and val == 0:
                i += 1

            elif i > 1 and val != 0:
                return False

        return True

    def to_continuous(_list):
        for i, val in enumerate(_list[::-1]):
            if val == 0:
                index = i
                break

        pre = _list[-index:]
        del _list[-index:]
        _list = pre + _list

        return (_list, index)

    def func(_x, a, b, c):
        return a * np.exp(-(_x - b)**2 / c)

    def gaussian_fit(_data):
        _data = list(_data)
        x = np.arange(len(_data))
        if is_continuous(_data) is True:
            try:
                popt, pcov = curve_fit(func, x, _data, bounds=([0.1, 0.1, 0.1], [1000, len(_data), 1000]))
            except RuntimeError:
                popt, pcov = curve_fit(func, x, _data, bounds=([0.1, 0.1, 0.1], [1000, len(_data), 200]))
        else:
            _data, moving_num = to_continuous(_data)
            try:
                popt, pcov = curve_fit(func, x, _data, bounds=([0.1, 0.1, 0.1], [1000, len(_data), 1000]))
            except RuntimeError:
                popt, pcov = curve_fit(func, x, _data, bounds=([0.1, 0.1, 0.1], [1000, len(_data), 200]))
            popt[1] -= moving_num
            popt[1] %= len(_data)
            if popt[1] >= (len(_data) - 0.5):
                popt[1] -= len(_data)

        return popt

    with open(CUR_DIR + 'gau.dat', 'wt') as fout:
        for i in range(0, len(fr), step):
            current_fr = fr.iloc[i, :]
            if all(val == 0 for val in current_fr):
                print(i, "0.0 NaN NaN", file=fout)
            elif all(val < 10 for val in current_fr) or (np.count_nonzero(current_fr) == 2):
                print(i, "NaN NaN NaN", file=fout)
            else:
                popt = gaussian_fit(current_fr).round(5)
                print(i, popt[0], popt[1], popt[2], file=fout)

def linear_fitting(arr):
    x = range(len(arr))
    y = arr
    model = LinearRegression()
    model.fit(np.c_[x], np.c_[y])
    r_square = model.score(np.c_[x], np.c_[y])
    y_pred = model.predict(np.c_[x])
    # # get equation from fitting (y = ax + b)
    # coe = np.c_[model.coef_, model.intercept_][0]
    # eq = np.poly1d(coe)
    #
    # # comparision of data and fitting result
    # y = eq(x)
    # plt.plot(x, arr)
    # plt.plot(x, y)
    # plt.show()

    return np.c_[x], y_pred  # (r_square, eq)

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


def evaluate_bump_rotation(gau: pd.DataFrame):
    segment_1st = gau.iloc[100:150, 2].values  # 10~15s
    segment_2nd = gau.iloc[150:200, 2].values  # 15~20s

    segment_1st_corrected = correct_rotation(segment_1st)
    segment_2nd_corrected = correct_rotation(segment_2nd)

    x1, y1 = linear_fitting(segment_1st_corrected)
    x2, y2 = linear_fitting(segment_2nd_corrected)
    
    # r_square_1st = linear_fitting(segment_1st_corrected)
    # r_square_2nd = linear_fitting(segment_2nd_corrected)

    return x1, y1, x2, y2


def plot_EB_fr(file_path, circuit='original', plot_gau=False, continuous=False):
    # use FiringRateALL.dat from Flysim

    fr = pd.read_csv(file_path, delim_whitespace=True, header=None)
    fr = fr.iloc[:, 1:19]  # EIP0 - EIP17
    fr.columns = np.arange(len(fr.columns))

    # eb_fr: plot EB region(avg of EIP)
    eb_fr = eip_to_eb(fr, circuit=circuit)
    fig = plt.figure(figsize=(5.5, 13))
    ax = fig.add_subplot(1, 2, 2)
    ax = plt.subplot2grid((1, 9), (0, 1), colspan=8)
    heatmap = ax.imshow(eb_fr, cmap='Blues', aspect=(75 / len(eb_fr)))

    if plot_gau is True:
        gau_fit(pd.DataFrame(eb_fr), step=10)
        gau = pd.read_csv(CUR_DIR + 'gau.dat', delim_whitespace=True, header=None)
        gau.dropna(inplace=True)

        for i in range(len(eb_fr)):
            if gau.iat[i, 1] > 5:
                index = i
                break

        if continuous is True:
            ax.plot(gau[2][index:], gau[0][index:], 'r', linewidth=1.5)

        elif continuous is False:
            last_gau = index
            next_gau = index
            while(next_gau < len(gau[0]) - 1):
                if abs(gau.iat[next_gau, 2] - gau.iat[next_gau+1, 2]) > 11:
                    ax.plot(gau[2][last_gau:next_gau], gau[0][last_gau:next_gau], 'firebrick', linewidth=10)
                    next_gau += 2
                    last_gau = next_gau
                else:
                    next_gau += 1
            ax.plot(gau[2][last_gau:], gau[0][last_gau:], 'firebrick', linewidth=10)

    plt.colorbar(heatmap)
    plt.xlabel('EB Region')
    plt.ylabel('Time(s)')
    ax.set_xlim([-0.5, 15.5])
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks([0, 4, 11, 15])
    ax.set_xticklabels(['R8', '4', '4', 'L8'])
    ax.set_yticklabels(map(int, ax.get_yticks() // 100))
    plt.savefig(os.path.splitext(file_path)[0]+'.png', format='png', bbox_inches='tight', dpi=100)
    # plt.savefig('DeltatoEPG_'+sys.argv[1]+'.png', format='png', bbox_inches='tight', dpi=100)


def eip_to_eb(eip_fr, circuit='original'):
    eb_fr = np.zeros((len(eip_fr), 16))

    if circuit == 'original':
        eip_rg = np.array([
            [1, 9, 0, 8],
            [1, 9, 10],
            [2, 10, 1],
            [2, 10, 11],
            [3, 11, 2],
            [3, 11, 12],
            [4, 12, 3],
            [4, 12, 13],
            [5, 13, 4],
            [5, 13, 14],
            [6, 14, 5],
            [6, 14, 15],
            [7, 15, 6],
            [7, 15, 16],
            [8, 16, 7],
            [8, 16, 17, 9]])
    elif circuit == 'symmetry':
        eip_rg = np.array([
            [0, 1, 17],
            [1, 17, 10],
            [1, 2, 10],
            [2, 10, 11],
            [2, 3, 11],
            [3, 11, 12],
            [3, 4, 12],
            [4, 12, 13],
            [4, 5, 13],
            [5, 13, 14],
            [5, 6, 14],
            [6, 14, 15],
            [6, 7, 15],
            [7, 15, 16],
            [7, 0, 16],
            [0, 16, 17]])

    for i in np.arange(len(eip_fr)):
        for j in np.arange(16):
            _sum = 0
            for n in eip_rg[j]:
                _sum += eip_fr.iat[i, n]
            eb_fr[i][j] = _sum / len(eip_rg[j])

    return pd.DataFrame(eb_fr)


if __name__ == "__main__":
    CUR_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'

    # fr = pd.read_csv(CUR_DIR+'FiringRateALL.dat', delim_whitespace=True, header=None)
    plot_EB_fr(CUR_DIR+'FiringRateALL.dat', circuit='symmetry', plot_gau=True, continuous=False)

