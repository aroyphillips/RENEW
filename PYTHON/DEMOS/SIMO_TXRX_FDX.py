#!/usr/bin/python
"""
    NOTE:::: NEED To rewrite below if this works.
    SIMO_TXRX_FDX.py

    NOTE: IRIS BOARDS MUST BE CHAINED FOR THIS SCRIPT TO WORK.
    ORDER MATTERS; FIRST BOARD (SERIAL1) IS THE ONE SENDING THE TRIGGER.
    TESTED WITH BOTH BOARDS USING BASE STATION ROOTFS IMAGE (NOT UE) AND
    ONLY FIRST BOARD CONNECTED TO HOST VIA ETHERNET

    This script is useful for testing the TDD operation.
    It programs two Irises in TDD mode with the following framing
    schedule:
        Iris 1: PGRG
        Iris 2: RGPG

    where P means a pilot or a pre-loaded signal, G means Guard
    band (no Tx or Rx action), R means Rx, and T means Tx,
    though not used in this script.

    The above determines the operation for each frame and each
    letter determines one symbol. Although every 16 consecutive
    frames can be scheduled separately.
    The pilot signal in this case is a sinusoid which is written
    into FPGA buffer (TX_RAM_A & TX_RAM_B for channels A & B)
    before the start trigger.

    The script programs the Irises in a one-shot mode, i.e.
    they run for just one frame. This means that each frame starts
    with a separate trigger. After the end of the frame,
    the script plots the two Rx symbols which are supposedly
    what each of the Iris boards received from each other (as shown
    in the schedule above).

    NOTE ON GAINS:
    Gain settings will vary depending on RF frontend board being used
    If using CBRS:
    rxgain: at 2.5GHz [3:1:105], at 3.6GHz [3:1:102]
    txgain: at 2.5GHz [16:1:93], at 3.6GHz [15:1:102]

    If using only Dev Board:
    rxgain: at both frequency bands [0:1:30]
    txgain: at both frequency bands [0:1:42]

    The code assumes both TX and RX have the same type of RF frontend board.

    Example:
        python3 SIMO_TXRX_FDX.py --serial1="RF3C000042" --serial2="RF3C000025"

---------------------------------------------------------------------
 Copyright © 2018-2019. Rice University.
 RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license
---------------------------------------------------------------------
"""

import sys

sys.path.append('../IrisUtils/')

import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
from optparse import OptionParser
import numpy as np
import time
import os
import math
import json
import matplotlib.pyplot as plt
from type_conv import *
from macros import *


#########################################
#              Functions                #
#########################################

def extract_ant(data: np.ndarray, both_channels: bool) -> np.ndarray:
    """
    Takes in 3D (num_sdrs*num_sdrs*2) matrix and spits out 2D (num_ant*num_ant) matrix
    :type data: np.ndarray
    :return: 2D matrix, where the indices reference the antennae TX/RX, rather than the board (s.t. ant. b on board n would be index 2n+1)
    """
    print("dtype of incoming data: {}".format(data.dtype))
    if not both_channels:
        data = np.squeeze(data)
        ant = 1
    else:
        ant = 2
    print("shape: {}".format(data.shape))
    num_ant = data.shape[0]
    retMatrix = np.zeros([num_ant, num_ant], "float32")

    for r, row in enumerate(data):
        retMatrix[r,:] = row.reshape(1, num_ant)

    return retMatrix

def simo_fdx_burst(hub, bserials, rate, freq, txgain, rxgain, numSamps, prefix_pad, postfix_pad, both_channels) -> np.array:
    # ## hub and multi-sdr integration copied from WB_CAL_DEMO.py
    if hub != "":
        hub_dev = SoapySDR.Device(dict(driver="remote", serial=hub))
    sdrs = [SoapySDR.Device(dict(driver="iris", serial=serial)) for serial in bserials]

    ant = 2 if both_channels else 1
    num_sdrs = len(sdrs)
    num_ants = num_sdrs * ant

    # assume trig_dev is part of the sdr nodes if no hub given
    trig_dev = None
    if hub != "":
        trig_dev = hub_dev
    else:
        trig_dev = sdrs[0]

    # set params on both channels
    for sdr in sdrs:
        info = sdr.getHardwareInfo()
        print("%s settings on device" % (info["frontend"]))
        for ch in [0, 1]:
            sdr.setBandwidth(SOAPY_SDR_TX, ch, 2.5 * rate)
            sdr.setBandwidth(SOAPY_SDR_RX, ch, 2.5 * rate)
            sdr.setSampleRate(SOAPY_SDR_TX, ch, rate)
            sdr.setSampleRate(SOAPY_SDR_RX, ch, rate)
            sdr.setFrequency(SOAPY_SDR_TX, ch, 'RF', freq - .75 * rate)
            sdr.setFrequency(SOAPY_SDR_RX, ch, 'RF', freq - .75 * rate)
            sdr.setFrequency(SOAPY_SDR_TX, ch, 'BB', .75 * rate)
            sdr.setFrequency(SOAPY_SDR_RX, ch, 'BB', .75 * rate)

            sdr.setGain(SOAPY_SDR_TX, ch, txgain)
            sdr.setGain(SOAPY_SDR_RX, ch, rxgain)

            sdr.setAntenna(SOAPY_SDR_RX, ch, "TRX")
            sdr.setDCOffsetMode(SOAPY_SDR_RX, ch, True)

            # Read initial gain settings
            read_lna = sdr.getGain(SOAPY_SDR_RX, 0, 'LNA')
            read_tia = sdr.getGain(SOAPY_SDR_RX, 0, 'TIA')
            read_pga = sdr.getGain(SOAPY_SDR_RX, 0, 'PGA')
            print("INITIAL GAIN - LNA: {}, \t TIA:{}, \t PGA:{}".format(read_lna, read_tia, read_pga))

            # gain setting from SISO_TXRX_TDD.py
            if "CBRS" in info["frontend"]:
                # Set gains to high val (initially)
                # sdr.setGain(SOAPY_SDR_TX, ch, txgain)  # txgain: at 2.5GHz [16:1:93], at 3.6GHz [15:1:102]
                # sdr.setGain(SOAPY_SDR_RX, ch, rxgain)  # rxgain: at 2.5GHz [3:1:105], at 3.6GHz [3:1:102]
                # else:
                # No CBRS board gains, only changing LMS7 gains
                sdr.setGain(SOAPY_SDR_TX, ch, "PAD", txgain)  # [0:1:42] txgain
                sdr.setGain(SOAPY_SDR_TX, ch, "ATTN", -6)
                sdr.setGain(SOAPY_SDR_RX, ch, "LNA", rxgain)  # [0:1:30] rxgain
                sdr.setGain(SOAPY_SDR_RX, ch, "LNA2", 14)
                sdr.setGain(SOAPY_SDR_RX, ch, "ATTN", 0 if freq > 3e9 else -18)

        # for ch in [0, 1]:
        #    if calibrate:
        #        sdr.writeSetting(SOAPY_SDR_RX, ch, "CALIBRATE", 'SKLK')
        #        sdr.writeSetting(SOAPY_SDR_TX, ch, "CALIBRATE", '')

        sdr.writeSetting("RESET_DATA_LOGIC", "")
        if not both_channels:
            sdr.writeSetting(SOAPY_SDR_RX, 1, 'ENABLE_CHANNEL', 'false')
            sdr.writeSetting(SOAPY_SDR_TX, 1, 'ENABLE_CHANNEL', 'false')

    trig_dev.writeSetting("SYNC_DELAYS", "")

    # Packet size
    symSamp = numSamps + prefix_pad + postfix_pad
    print("numSamps = %d" % numSamps)
    print("symSamps = %d" % symSamp)

    # Generate sinusoid to be TX
    Ts = 1 / rate
    s_freq = 1e5
    s_time_vals = np.array(np.arange(0, numSamps)).transpose() * Ts
    pilot = np.exp(s_time_vals * 1j * 2 * np.pi * s_freq).astype(np.complex64) * 1
    pad1 = np.array([0] * prefix_pad, np.complex64)
    pad2 = np.array([0] * postfix_pad, np.complex64)
    wbz = np.array([0] * symSamp, np.complex64)
    pilot1 = np.concatenate([pad1, pilot, pad2])
    pilot2 = wbz

    # pilot1_energy = np.sum(np.abs(cfloat2uint32(pilot1, order='QI')**2))/len(pilot1)


    # Initialize RX Matrix | num_sdrs x num_sdrs filled with None, to be filled with np.ndarray of np.ndarray of np.ndarray of np.uint32
    # The layers ought to be: array of each iteration (distinct TX board), each of which is an array of each board's RX,
    # each of which is an array of two arrays (one per antenna), each antenna having an array of values

    # this initializes all values to be None, might cause problems if something isn't overwritten for some reason
    rxMatrix = np.empty([num_ants, num_sdrs, ant], np.ndarray)

    # Create RX streams
    # CS16 makes sure the 4-bit lsb are samples are being sent

    rxStreams = [sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1]) for sdr in sdrs]


    # Set Schedule #NOTE: this config is still unchanged from SISO_TXRX_TDD.py, needs to be fixed

    txsched = "PG"
    rxsched = "RG"
    print("Node 1 schedule %s " % txsched)
    print("Node 2 schedule %s " % rxsched)
    # Send one frame (set mamx_frame to 1)
    txconf = {"tdd_enabled": True,
              "frame_mode": "free_running",
              "symbol_size": symSamp,
              "frames": [txsched],
              "max_frame": 1}
    rxconf = {"tdd_enabled": True,
              "frame_mode": "free_running",
              "dual_pilot": False,
              "symbol_size": symSamp,
              "frames": [rxsched],
              "max_frame": 1}

    # SW Delays
    for sdr in sdrs:
        sdr.writeSetting("TX_SW_DELAY", str(30))

        # TDD_MODE Setting
        sdr.writeSetting("TDD_MODE", "true")

    #for sdr in sdrs:
    #    sdr.writeRegisters("TX_RAM_B", 0, cfloat2uint32(pilot2, order='QI').tolist())

    # Average Energy in signal
    energy_func = lambda x: np.sum(np.abs(x)**2)/len(x)
    dB_func = lambda x: 10*np.log10(x)

    # implement loop logic here to make each board the TX board once per antenna, run num_ant times
    for i, txsdr in enumerate(sdrs):
        # repeat twice if both channels active
        for tx_ant in range(ant):
            activate_B = tx_ant % 2
            if activate_B:
                # if second time through, activate channel B
                txsdr.writeRegisters("TX_RAM_B", 0, cfloat2uint32(pilot1, order='QI').tolist())
                txsdr.writeRegisters("TX_RAM_A", 0, cfloat2uint32(pilot2, order='QI').tolist())
            else:
                # activate channel A first time through
                txsdr.writeRegisters("TX_RAM_A", 0, cfloat2uint32(pilot1, order='QI').tolist())
                txsdr.writeRegisters("TX_RAM_B", 0, cfloat2uint32(pilot2, order='QI').tolist())

            txsdr.writeSetting("TDD_CONFIG", json.dumps(txconf))

            # list comp. here makes sure we loop over every other board
            for j, rxsdr in enumerate([board for board in sdrs if board != txsdr]):
                rxsdr.writeSetting("TDD_CONFIG", json.dumps(rxconf))

            # build rx Arrays
            rxArray_single_channel = np.array([0]*symSamp, np.complex64)
            rxArray_both_channels = np.array([rxArray_single_channel]*2)
            rxArrays = np.array([rxArray_both_channels]*num_sdrs)

            flags = 0
            rList = np.empty(num_sdrs, object)

            # activate streams
            for r, sdr in enumerate(sdrs):
                rList[r] = sdr.activateStream(rxStreams[r], flags, 0)
                if rList[r] < 0:
                    print("Problem activating stream # %d" % i)
            trig_dev.writeSetting("TRIGGER_GEN", "")

            dBArrays = np.zeros([num_sdrs, ant]) #NOTE: this will initialize it to be 0s, which might be problematic

            # read Streams
            for r, sdr in enumerate(sdrs):
                rList[r] = sdrs[r].readStream(rxStreams[r], rxArrays[r], symSamp)
                print("reading stream #{} ({})".format(r, rList[r]))

                # read num_ant stream (either A or B depending) on which one sent
                for rx_ant in range(ant):
                    dBArrays[r][rx_ant] = dB_func(energy_func(rxArrays[r][rx_ant]))
                    amp = np.max(abs(rxArrays[r][rx_ant]))
                    if amp > 0.1:
                        print("Board {0} with board {1}, max amp:{2}".format(i*2+tx_ant,r*2+rx_ant, amp))


            for r, sdr in enumerate(sdrs):
                sdr.deactivateStream(rxStreams[r])
    #        print("\n ================ \n ")
    #        print(dBArrays.shape)
    #        print(dBArrays)
    #        print("\n =============== \n")
            rxMatrix[i*ant+tx_ant] = dBArrays
#    print(np.transpose(rxMatrix))

    # End of data collection loops #

    # ADC_rst, stops the tdd time counters, makes sure next time runs in a clean slate
    tdd_conf = {"tdd_enabled": False}
    for sdr in sdrs:
        sdr.writeSetting("RESET_DATA_LOGIC", "")
        sdr.writeSetting("TDD_CONFIG", json.dumps(tdd_conf))
        sdr.writeSetting("TDD_MODE", "false")

    for r, sdr in enumerate(sdrs):
        sdr.deactivateStream(rxStreams[r])
        sdr.closeStream(rxStreams[r])
        sdrs[r] = None
    print("3D dB Matrix: \n", rxMatrix)
    retMatrix = extract_ant(rxMatrix, both_channels)
    retMatrix = np.transpose(retMatrix)
    print("2D dB Matrix: \n", retMatrix)
    return(retMatrix)


#########################################
#                  Main                 #
#########################################
def main():
    parser = OptionParser()
    parser.add_option("--bnodes", type="string", dest="bnodes", help="file name containing serials on the base station",
                      default="../IrisUtils/data_in/stadium_serials.txt")
    parser.add_option("--hub", type="string", dest="hub", help="Hub node", default="FH4A000002")
    parser.add_option("--rate", type="float", dest="rate", help="Tx sample rate", default=5e6)
    parser.add_option("--txgain", type="float", dest="txgain", help="Tx gain (dB)",
                      default=25.0)  # See documentation at top of file for info on gain range
    parser.add_option("--rxgain", type="float", dest="rxgain", help="Rx gain (dB)",
                      default=20.0)  # See documentation at top of file for info on gain range
    parser.add_option("--freq", type="float", dest="freq", help="Optional Tx freq (Hz)", default=3.6e9)
    parser.add_option("--numSamps", type="int", dest="numSamps", help="Num samples to receive", default=512)
    parser.add_option("--prefix-pad", type="int", dest="prefix_length",
                      help="prefix padding length for beacon and pilot", default=82)
    parser.add_option("--postfix-pad", type="int", dest="postfix_length",
                      help="postfix padding length for beacon and pilot", default=68)
    parser.add_option("--both-channels", action="store_true", dest="both_channels", help="transmit from both channels",
                      default=False)
    parser.add_option("--output", type="str", dest="output",
                      help="output destination file", default="data_out/SIMO_out.txt")
    (options, args) = parser.parse_args()

    bserials = []
    with open(options.bnodes, "r") as f:
        for line in f.read().split():
            if line[0] != '#':
                bserials.append(line)
            else:
                continue

    data = simo_fdx_burst(
        hub=options.hub,
        bserials=bserials,
        rate=options.rate,
        freq=options.freq,
        txgain=options.txgain,
        rxgain=options.rxgain,
        numSamps=options.numSamps,
        prefix_pad=options.prefix_length,
        postfix_pad=options.postfix_length,
        both_channels=options.both_channels,
    )
    data.dump(options.output)

if __name__ == '__main__':
    main()

