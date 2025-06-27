import numpy as np
from scipy.stats import entropy

def _entropy(freq, D):
    num_class = freq.shape[0]
    class_margin_freq = []
    for i in range(num_class):
        class_margin_freq.append(float(np.sum(freq[i]) + 1e-10))
    class_margin_freq = np.array(class_margin_freq)
    num_freq = np.sum(class_margin_freq)

    class_margin_D = []
    for i in range(num_class):
        class_margin_D.append(float(np.sum(D[i]) + 1e-10))
    class_margin_D = np.array(class_margin_D)
    num_D = np.sum(class_margin_D)

    score = entropy(class_margin_freq / sum(class_margin_freq))
    score = float(score * num_freq / num_D)
    return score

def _chisquare(freq, D):
    num_class = freq.shape[0]
    class_margin_freq = []
    for i in range(num_class):
        class_margin_freq.append(float(np.sum(freq[i]) + 1e-10))
    class_margin_freq = np.array(class_margin_freq)
    num_freq = sum(class_margin_freq)

    class_margin_D = []
    for i in range(num_class):
        class_margin_D.append(float(np.sum(D[i]) + 1e-10))
    class_margin_D = np.array(class_margin_D)
    num_D = sum(class_margin_D)

    score = 0.0
    for i in range(num_class):
        tmp = (float(class_margin_freq[i] - num_freq * class_margin_D[i] / num_D)) ** 2
        tmp = tmp / float(num_freq * class_margin_D[i])
        score += tmp
    return score

def _mi(freq, D):
    num_class = freq.shape[0]
    class_margin_freq = []
    for i in range(num_class):
        class_margin_freq.append(float(np.sum(freq[i]) + 1e-10))
    class_margin_freq = np.array(class_margin_freq)
    num_freq = sum(class_margin_freq)

    class_margin_D = []
    for i in range(num_class):
        class_margin_D.append(float(np.sum(D[i]) + 1e-10))
    class_margin_D = np.array(class_margin_D)
    num_D = sum(class_margin_D)

    score = 0.0
    for i in range(num_class):
        inlog = float(num_D * class_margin_freq[i] / (num_freq * class_margin_D[i]))
        tmp = float(class_margin_freq[i] * np.log2(inlog))
        score += tmp
    score = score / num_D
    return score

def getMetric(freq, D, metric = 'entropy'):
    if D is None:
        print("Full feature frequency required!")
        return -1
    if metric == 'entropy':
        return _entropy(freq, D)
    if metric == 'chi2':
        return _chisquare(freq, D)
    if metric == 'mi':
        return _mi(freq, D)