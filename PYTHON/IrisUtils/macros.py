# -*- coding: UTF-8 -*-

"""
 macros.py

 Mainly for register setup

---------------------------------------------------------------------
 Copyright © 2018-2019. Rice University.
 RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license
---------------------------------------------------------------------
"""

# Reset
RF_RST_REG = 48

# AGC registers Set
FPGA_IRIS030_WR_AGC_ENABLE_FLAG = 232
FPGA_IRIS030_WR_AGC_RESET_FLAG = 236
FPGA_IRIS030_WR_IQ_THRESH = 240
FPGA_IRIS030_WR_NUM_SAMPS_SAT = 244
FPGA_IRIS030_WR_MAX_NUM_SAMPS_AGC = 248
FPGA_IRIS030_WR_RSSI_TARGET = 252
FPGA_IRIS030_WR_WAIT_COUNT_THRESH = 256
FPGA_IRIS030_WR_AGC_SMALL_JUMP = 260
FPGA_IRIS030_WR_AGC_BIG_JUMP = 264
FPGA_IRIS030_WR_AGC_TEST_GAIN_SETTINGS = 268
FPGA_IRIS030_WR_AGC_LNA_IN = 272
FPGA_IRIS030_WR_AGC_TIA_IN = 276
FPGA_IRIS030_WR_AGC_PGA_IN = 280

# RSSI register Set
FPGA_IRIS030_RD_MEASURED_RSSI = 284

# Packet Detect Register Set
FPGA_IRIS030_WR_PKT_DET_THRESH = 288
FPGA_IRIS030_WR_PKT_DET_NUM_SAMPS = 292
FPGA_IRIS030_WR_PKT_DET_ENABLE = 296
FPGA_IRIS030_WR_PKT_DET_NEW_FRAME = 300

# CBRS Gains
FPGA_IRIS030_WR_AGC_ATTN_IN = 304
FPGA_IRIS030_WR_AGC_LNA1_IN = 308
FPGA_IRIS030_WR_AGC_LNA2_IN = 312

# Init gain setting
FPGA_IRIS030_WR_AGC_GAIN_INIT = 316

# AGC Cont'd
FPGA_IRIS030_WR_NUM_SAT_STAGES = 320
FPGA_IRIS030_WR_NUM_FINE_STAGES = 324

TX_GAIN_CTRL = 88