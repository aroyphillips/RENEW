#!/usr/bin/python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
from optparse import OptionParser


def heatmapPlot(data: np.ndarray) -> None:
    """

    :param data: 2D square numpy array with numerical values (square matrix)
    :return:
    """

    if data.shape[0] == data.shape[1]:
        numAnt = data.shape[0]
        print("Starting plot with {} antennae".format(numAnt))
    else:
        raise RuntimeError("The matrix passed is not square.")

    if data.dtype != "float32":
        print("Check dtype, changing to float32")
        data.astype("float32")

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap="jet", interpolation="nearest")

    for edge, spine in ax.spines.items():
        spine.set_visible(True)  # If this is True, there is a border around the heatmap

    ax.tick_params(which="minor", bottom=False, left=False)
    ax.grid(which="both", color="black", linestyle='-', linewidth=.2, alpha=1.0)

    if data.shape[0] > 15:

        ax.set_xticklabels(np.arange(data.shape[1]+1, step=5))
        ax.set_yticklabels(np.arange(data.shape[1]+1, step=5))

        # show every fifth
        ax.set_xticks(np.arange(start=0, stop=data.shape[1]+1, step=5)-.5)

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_yticks(np.arange(start=0, stop=data.shape[0]+1, step=5)-.5)

        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    else:
        ax.set_xticklabels(np.arange(data.shape[1]))
        ax.set_yticklabels(np.arange(data.shape[0]))
        ax.set_xticks(np.arange(data.shape[1]+1)-.5)
#        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_yticks(np.arange(data.shape[0]+1)-.5)
#        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(which="minor", bottom=False, left=False)

    plt.setp(ax.get_xticklabels(), ha="center", rotation_mode="anchor")
    ax.invert_yaxis()

    cbarlabel = "Coupling Strength (dB)"
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=90, va="top")

    ax.set_title("Self-Interference Measure")
    fig.tight_layout()
    plt.xlabel("TX Ant. Index")
    plt.ylabel("RX Ant. Index")
    plt.show()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--dataFile", type="string", dest="dataFile", help="file name containing data dumped from SIMO_TXRX_FDX.py",
                      default="data_out/SIMO_out.txt")
    (options, args) = parser.parse_args()
    data = np.load(options.dataFile)
    heatmapPlot(data)
