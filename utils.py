import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def visualize(viz, xtitle, ytitle, title, show=False, dir=None, legend=True):
    fig, axes = plt.subplots()
    for v in viz:
        axes.plot(np.arange(1, v[0].shape[0] + 1), v[0], label=v[1])
    if legend:
        axes.legend()
    axes.set_xlabel(xtitle)
    axes.set_ylabel(ytitle)
    axes.set_title(title)
    if dir is not None:
        plt.savefig(dir)
    if show:
        plt.show()


def cleanseNA(fea_vec):
    avail_index = np.argwhere(~np.isnan(fea_vec)).flatten().tolist()
    return fea_vec[avail_index]


def makeVal(df, feature):
    df[feature] = df[feature].astype(float)
    val = df[feature].unique()
    val = np.sort(val)
    val = cleanseNA(val)
    return val


def makePrebins(df, feature, label, num_classes=2):
    # Prepare discretized values
    df[feature] = df[feature].astype(float)
    val = df[feature].unique()
    val = np.sort(val)
    val = cleanseNA(val)

    catcode = pd.Series(pd.Categorical(df[label], categories=df[label].unique())).cat.codes
    num_classes = max(catcode) + 1

    valdict = []
    for i in range(num_classes):
        valdict.append(dict.fromkeys(val, 0))

    for i in range(len(df)):
        if np.isnan(df[feature][i]) or catcode[i] == -1:
            continue
        valdict[catcode[i]][df[feature][i]] += 1

    freq = []
    for i in range(num_classes):
        freq.append([0] + list(valdict[i].values()))

    freq = np.array(freq)

    return val, freq, valdict


def initMode(mode, val=1e9):
    if mode == 'min':
        return val
    else:
        return -val


def binSearch(arr, val):
    L = 0
    R = len(arr) - 1
    while (L <= R):
        mid = int((L + R) / 2)
        if arr[mid] == val:
            return mid
        if arr[mid] < val:
            L = mid + 1
        else:
            R = mid - 1
    return -1


def discretizeFeat(df, fea, split):
    full_split = [df[fea].min() - 1] + list(split) + [
        df[fea].max() + 1]
    return pd.cut(df[fea], bins=full_split, labels=False)


def mask2Split(mask, val):
    splt = mask * val
    return splt[splt != 0]


def split2Mask(split, val):
    splt_ptr = 0
    mask = np.zeros_like(val).astype(int)

    for i in range(len(val)):
        if splt_ptr < len(split) and math.fabs(val[i] - split[splt_ptr]) < 1e-3:
            mask[i] = 1
            splt_ptr += 1
        else:
            mask[i] = 0

    return mask