import numpy as np

"""
scan_options ={
"l" : length,
'x' : exclude,
'c' : column,
'b' : partitions,
'D' : corrlength,
'V' : verbosity,
'o' : file_out
}
"""


def cond_entropy(bins, t, partitions):
    """
    :param bins: The discrete version fo the times series
    :type bins: array of ints
    :param t: the time delay
    :type t: int
    :param partitions: The number of bins used for discretization
    :type partitions: int
    :return: conditional entropy (mutual)
    :rtype: float
    """
    count = 0
    cond_ent = 0.0
    h1 = np.zeros(partitions)
    h11 = h1.copy()
    h2 = np.zeros((partitions, partitions))

    for i in range(t, len(bins)):
        hii = bins[i]
        hi = bins[i - t]
        h1[hi] += 1
        h11[hii] += 1
        h2[hi][hii] += 1
        count += 1

    norm = 1.0 / count
    cond_ent = 0.0

    for i in range(partitions):
        hpi = h1[i] * norm
        if (hpi > 0.0):
            for j in range(partitions):
                hpj = h11[j] * norm
                if (hpj > 0.0):
                    pij = h2[i][j] * norm
                    if (pij > 0.0):
                        cond_ent += pij * np.log(pij / hpj / hpi)
    return cond_ent


def mutual(series, corrlength, partitions):
    """

    :param series: array of the time series for times : 0,delta, 2 delta....
    :type series: numpy array
    :param corrlength: maximal time delay
    :type corrlength: int
    :param partitions: number of bins to discretize the data for MI
    :type partitions: int
    :return: array size corrlength +1 ( 0 : shanon entropy)
    :rtype: numpy array of floats
    """
    length = series.shape[0]

    # Rescaling Data
    mn = series.min()
    interval = series.max() - mn
    if interval == 0:
        raise "Constant data"
    series = (series - mn) / interval

    bins = np.zeros(length)
    bins = np.clip((series * partitions).astype(int), 0, partitions - 1)

    if (corrlength >= length):
        corrlength = length - 1
    ent = []
    for tau in range(corrlength + 1):
        ent.append(cond_entropy(bins, tau, partitions))

    return np.array(ent)
