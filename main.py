import math
from collections import Counter
import numpy as np
from numpy import *
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import utils
from binalgo import scoreDP

def mutual_information(X, Y):
    counter_X = Counter(X)
    counter_Y = Counter(Y)

    N = len(X)

    P_X = {x: count / N for x, count in counter_X.items()}
    P_Y = {y: count / N for y, count in counter_Y.items()}

    joint_counts = Counter(zip(X, Y))
    P_XY = {key: count / N for key, count in joint_counts.items()}

    MI = 0
    for (x, y), p_xy in P_XY.items():
        if p_xy > 0:
            MI += p_xy * math.log2(p_xy / (P_X[x] * P_Y[y]))

    return MI

def compute_prob(X, Y, Z):
    n = len(Z)
    P_z = Counter(Z)
    for k in P_z:
        P_z[k] /= n

    P_xz = {}
    P_yz = {}
    P_xyz = {}

    for x, y, z in zip(X, Y, Z):
        if (x, z) not in P_xz:
            P_xz[(x, z)] = 0
        P_xz[(x, z)] += 1

        if (y, z) not in P_yz:
            P_yz[(y, z)] = 0
        P_yz[(y, z)] += 1

        if (x, y, z) not in P_xyz:
            P_xyz[(x, y, z)] = 0
        P_xyz[(x, y, z)] += 1

    for (x, z) in P_xz:
        P_xz[(x, z)] /= Counter(Z)[z]
    for (y, z) in P_yz:
        P_yz[(y, z)] /= Counter(Z)[z]
    for (x, y, z) in P_xyz:
        P_xyz[(x, y, z)] /= Counter(Z)[z]

    return P_z, P_xz, P_yz, P_xyz

def CMI(X, Y, Z):
    P_z, P_xz, P_yz, P_xyz = compute_prob(X, Y, Z)
    I_xyz = 0

    for (x, y, z) in P_xyz:
        p_xyz = P_xyz[(x, y, z)]
        p_xz = P_xz[(x, z)]
        p_yz = P_yz[(y, z)]
        p_z = P_z[z]

        if p_xyz > 0 and p_xz > 0 and p_yz > 0:
            I_xyz += p_z * p_xyz * np.log(p_xyz / (p_xz * p_yz))

    return I_xyz

def algorithm1(F, C):
    result = pd.concat([F, C], axis=1)
    Fc = np.zeros((F.shape[0], 1))
    Dc = []
    split_val = []
    Jc = []
    cnt = 0

    for i in F.columns:
        val, freq, _ = utils.makePrebins(result, i, C.columns[0])
        opt_score, spl_val, j = scoreDP(val, freq)
        Jrel = mutual_information(pd.cut(F[:][i], bins=[float('-inf')] + spl_val, labels=False), C['class'].T)
        if Jrel > 0:
            Fi = F[:][i].values
            Fi = Fi.reshape(-1, 1)
            Fc = insert(Fc, [cnt + 1], Fi, axis=1)
            Dc.append(j)
            Jc.append(Jrel)
            split_val.append(spl_val)
            cnt = cnt + 1
    Fc = np.delete(Fc, 0, axis=1)

    return Fc, Dc, Jc, split_val

def algorithm2(fi, C, di, S, split_val, Jjmi_pre, e):
    check = 0
    Jjmi = 0

    fi_discretized = pd.cut(fi, bins=[float('-inf')] + split_val, labels=False)
    Jjmi_ori = mutual_information(fi_discretized, C['class'].T)
    for i in range(0, len(S)):
        Jjmi_ori = Jjmi_ori + (CMI(fi_discretized, S[i], C['class'].T) - mutual_information(fi_discretized, S[i])) / len(S)
    if Jjmi_ori > Jjmi_pre:
        check = 1
    print(Jjmi_ori, Jjmi)

    return Jjmi_ori, check

def ADSM(F, C, e):
    Fc, Dc, Jc, split_val = algorithm1(F, C)
    Fc = Fc.T

    # Sort features based on their mutual information scores (Jc)
    sorted_indices = np.argsort(Jc)[::-1]
    Fc = [Fc[i] for i in sorted_indices]
    Dc = [Dc[i] for i in sorted_indices]
    Jc = [Jc[i] for i in sorted_indices]
    split_val = [split_val[i] for i in sorted_indices]

    S = []
    D = []
    S.append(pd.cut(Fc[0], bins=[float('-inf')] + split_val[0], labels=False))
    D.append(Dc[0])

    Fc.pop(0)
    Dc.pop(0)
    split_val.pop(0)

    T = -np.inf
    for i in range(len(Fc)):
        fi = Fc[i]
        di = Dc[i]
        Jjmi, check = algorithm2(fi, C, di, S, split_val[i], T, e)
        if check:
            S.append(pd.cut(Fc[i], bins=[float('-inf')] + split_val[i], labels=False))
            D.append(Dc[i])
            T = Jjmi

    S_ori = S[:]
    tmp = -np.inf
    min_position_tmp = -1
    print(len(S))
    while True:
        ep = np.zeros(len(S))
        for p in range(0, len(S)):
            ep[p] = mutual_information(S[p], C['class'].T)
            for i in range(0, len(S)):
                if i == p:
                    continue
                ep[p] = ep[p] + (CMI(S[p], S[i], C['class'].T) - mutual_information(S[p], S[i])) / (len(S) - 1)
        min_position = np.argmin(ep)
        if min_position_tmp == -1:
            min_position_tmp = min_position
            tmp = ep[min_position]
            S_ori = S
            S = S[:min_position_tmp] + S[min_position_tmp + 1:]
            continue
        print(ep[min_position], tmp + e / len(S), np.median(ep))
        if (np.median(ep)) > (tmp + e / len(S)):
            min_position_tmp = min_position
            tmp = ep[min_position]
            S_ori = S
            S = S[:min_position_tmp] + S[min_position_tmp + 1:]
        else:
            break
    S = S_ori
    print(len(S))
    return S, D

dataset = fetch_ucirepo(id=109)
X = dataset.data.features
y = dataset.data.targets

e = 25
S, D = ADSM(X, y, e)
y = y.values
X = np.array(S).T

clf = SVC(kernel='linear')
scores_selected = cross_val_score(clf, X, y, cv=10)
print(len(S))
print(f"Accuracy (10-CV): {scores_selected.mean():.4f}")