#!/usr/bin/python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
from optparse import OptionParser



def plotting(data: np.ndarray) -> None:
    """
    Takes in data file from softnull_partition.py and plots it, similarly to Fig. 8 in the SoftNull paper.
    Takes in an np.ndarray named data where data.shape = (5, 29) (for now)
    Row n is data[n], corresponding to a specific partition,
     and the row ordering is: Random, North-South, East-West, NW-SE, Interleaved
    if data[n] = array of None, will ignore that partitioning in plot
    :param data:
    :return: None
    """
    if data.shape[0] != 5:
        raise RuntimeError("The input matrix should have 5 rows")
    name_dict = {
        "0": "R",
        "1": "NS",
        "2": "EW",
        "3": "NWSE",
        "4": "I"
    } # type: dict
    abbr_dict = {
        "R": "Random",
        "NS": "North-South",
        "EW": "East-West",
        "NWSE": "NW-SE",
        "I": "Interleaved"
    } # type: dict
    data_dict = {} # type: dict
    legend = []
    symbols = ["k^", "rd", "bs", "yo", "gx"]
    x = np.linspace(4, 32, num=29)
    for i, row in enumerate(data):
        if row[0] is not None:
            data_dict[str(i)] = row
            legend.append(abbr_dict[name_dict[str(i)]])
    for r in data_dict.keys():
        plt.plot(x, data_dict[r], symbols[int(r)], ls='-', fillstyle="none")
    plt.legend(labels=legend)
    plt.show()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--dataFile", type="string", dest="dataFile", help="file name containing data dumped from SIMO_TXRX_FDX.py",
                      default="data_out/partitions_out.txt")
    (options, args) = parser.parse_args()
    data = np.load(options.dataFile)
    plotting(data)
