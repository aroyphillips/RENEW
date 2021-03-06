/*
 Helper functions for signal processing and communications
 Generate training sequence for pilots and preambles.

 Supports:
 STS - 802.11 Short training sequence. Generates one symbol, 16 complex I/Q samples.
 LTS - 802.11 Long training sequence. Generates 2.5 symbols, cp length of 32 samples, 
       for a total of 160 complex I/Q samples.
 LTE Zadoff Chu - Generates the 25th root length-63 Zadoff-Chu sequence.
       Total of 63-long complex IQ samples.
 Gold IFFT - Total of 128-long complex IQ samples including a 32-sample cyclic prefix
 Hadamard - Real valued sequence. Possible lenghts: {2, 4, 8, 16, 32, 64}

---------------------------------------------------------------------
 Copyright (c) 2018-2019, Rice University 
 RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license
 Author(s): Oscar Bejarano: obejarano@rice.edu
            Rahman Doost-Mohamamdy: doost@rice.edu
---------------------------------------------------------------------
*/

#include "include/comms-lib.h"
#include <queue>
//#include <itpp/itbase.h>

int CommsLib::findLTS(const std::vector<std::complex<double>>& iq, int seqLen)
{
    /*
     * Find 802.11-based LTS (Long Training Sequence)
     * Input:
     *     iq        - IQ complex samples (vector)
     *     seqLen    - Length of sequence
     * Output:
     *     best_peak - LTS peak index (correlation peak)
     */

    float lts_thresh = 0.8;
    std::vector<std::vector<double>> lts_seq;
    int best_peak;

    // Original LTS sequence
    lts_seq = CommsLib::getSequence(seqLen, LTS_SEQ);

    // Re-arrange into complex vector, flip, and compute conjugate
    std::vector<std::complex<double>> lts_sym(64);
    std::vector<std::complex<double>> lts_sym_conj(lts_sym.size());
    for (size_t i = 0; i < lts_sym.size(); i++) {
        // lts_seq is a 2x160 matrix (real/imag by seqLen=160 elements)
        // grab one symbol and flip around
        lts_sym[i] = std::complex<double>(lts_seq[0][seqLen - 1 - i], lts_seq[1][seqLen - 1 - i]);
        // conjugate
        lts_sym_conj[i] = std::conj(lts_sym[i]);
    }

    // Equivalent to numpy's sign function
    std::vector<std::complex<double>> iq_sign = CommsLib::csign(iq);

    // Convolution
    std::vector<double> lts_corr = CommsLib::convolve(iq_sign, lts_sym_conj);
    double lts_limit = lts_thresh * *std::max_element(lts_corr.begin(), lts_corr.end());

    // Find all peaks, and pairs that are lts_sym.size() samples apart
    std::queue<int> valid_peaks;
    for (size_t i = lts_sym.size(); i < lts_corr.size(); i++) {
        if (lts_corr[i] > lts_limit && lts_corr[i - lts_sym.size()] > lts_limit)
            valid_peaks.push(i - lts_sym.size());
    }

    // Use first LTS found
    if (valid_peaks.empty()) {
        best_peak = -1;
    } else {
        best_peak = valid_peaks.front();
    }

    return best_peak;
}

std::vector<std::complex<double>> CommsLib::csign(std::vector<std::complex<double>> iq)
{
    /*
     * Return element-wise indication of the sign of a number (for complex vector).
     *
     * For complex-valued inputs:
     *     sign(x.real) + 0j if x.real != 0 else sign(x.imag) + 0j
     *
     * where sign(x) is given by
     *     -1 if x < 0, 0 if x==0, 1 if x > 0
     */
    std::vector<std::complex<double>> iq_sign;
    for (int i = 0; i < static_cast<int>(iq.size()); i++) {
        // sign(x.real) + 0j if x.real != 0 else sign(x.imag) + 0j
        std::complex<double> x = iq[i];
        if (x.real() != 0) {
            iq_sign.push_back((x.real() > 0) ? 1 : (x.real() < 0) ? -1 : 0);
        } else {
            iq_sign.push_back((x.imag() > 0) ? 1 : (x.imag() < 0) ? -1 : 0);
        }
    }
    return iq_sign;
}

std::vector<double> CommsLib::convolve(std::vector<std::complex<double>> const& f, std::vector<std::complex<double>> const& g)
{
    /* Convolution of two vectors
     * Source:
     * https://stackoverflow.com/questions/24518989/how-to-perform-1-dimensional-valid-convolution
     */
    int const nf = f.size();
    int const ng = g.size();
    int const n = nf + ng - 1;
    std::vector<double> out(n, 0);
    std::vector<std::complex<double>> outc(n, 0);
    for (auto i(0); i < n; ++i) {
        int const jmn = (i >= ng - 1) ? i - (ng - 1) : 0;
        int const jmx = (i < nf - 1) ? i : nf - 1;
        for (auto j(jmn); j <= jmx; ++j) {
            outc[i] += f[j] * g[i - j];
        }
        out[i] += abs(outc[i]);
    }
    return out;
}

std::vector<float> CommsLib::magnitudeFFT(std::vector<std::complex<float>> const& samps, std::vector<float> const& win, size_t fftSize)
{
    std::vector<std::complex<float>> preFFT(samps.size());

    for (size_t n = 0; n < fftSize; n++) {
        preFFT[n] = samps[n] * win[n];
    }

    std::vector<std::complex<float>> fftSamps = CommsLib::FFT(preFFT, fftSize);

    // compute magnitudes
    std::vector<float> fftMag;
    fftMag.reserve(fftSize);
    for (size_t n = fftSize / 2; n < fftSize; n++) {
        fftMag.push_back(std::norm(fftSamps[n]));
    }
    for (size_t n = 0; n < fftSize / 2; n++) {
        fftMag.push_back(std::norm(fftSamps[n]));
    }
    std::reverse(fftMag.begin(), fftMag.end()); // not sure why we need reverse here, but this seems to give the right spectrum
    return fftMag;
}

// Take ffsSize samples of (1 - cos(x)) / 2 from 0 up to 2pi
std::vector<float> CommsLib::hannWindowFunction(size_t fftSize)
{
    std::vector<float> winFcn(1, 0);
    double step = 2 * M_PI / fftSize;

    // Compute the samples for the first half.
    for (size_t n = 1; n < fftSize / 2; n++) {
        winFcn.push_back((1 - std::cos(step * n)) / 2);
    }
    // If a sample lies at the center, just use (1-cos(pi))/2 == 1.
    if (fftSize % 2 == 0)
        winFcn.push_back(1);
    // The second half is a mirror image of the first, so just copy.
    for (size_t n = fftSize / 2 + 1; n < fftSize; n++)
        winFcn.push_back(winFcn[fftSize - n]);
    return winFcn;
}

double CommsLib::windowFunctionPower(std::vector<float> const& win)
{
    double windowPower = (0);
    size_t N = win.size();
    for (size_t n = 0; n < win.size(); n++) {
        windowPower += std::norm(win[n]);
    }
    windowPower = std::sqrt(windowPower / N);
    return 20 * std::log10(N * windowPower);
}

template <typename T>
T CommsLib::findTone(std::vector<T> const& magnitude, double winGain, double fftBin, size_t fftSize, const size_t delta)
{
    /*
     * Find the tone level at a specific interval in the input Power Spectrum
     * fftBins assumed interval is [-0.5, 0.5] which is coverted to [0, fftSize-1]
     */
    // make sure we don't exceed array bounds
    size_t first = std::max<size_t>(0, std::lround((fftBin + 0.5) * fftSize) - delta);
    size_t last = std::min<size_t>(fftSize - 1, std::lround((fftBin + 0.5) * fftSize) + delta);
    T refLevel = magnitude[last];
    for (size_t n = first; n < last; n++) {
        if (magnitude[n] > refLevel)
            refLevel = magnitude[n];
    }
    return 10 * std::max(std::log10(refLevel), (T)(-20.0)) - (T)winGain;
}

float CommsLib::measureTone(std::vector<std::complex<float>> const& samps, std::vector<float> const& win, double winGain, double fftBin, size_t fftSize, const size_t delta)
{
    return findTone(magnitudeFFT(samps, win, fftSize), winGain, fftBin, fftSize, delta);
}

std::vector<int> CommsLib::getDataSc(int fftSize)
{
    std::vector<int> data_sc;
    if (fftSize == 64) {
        int sc_ind[48] = { 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            22, 23, 24, 25, 26, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50,
            51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63 };
        data_sc.assign(sc_ind, sc_ind + 48);
    }
    return data_sc;
}

std::vector<int> CommsLib::getNullSc(int fftSize)
{
    std::vector<int> null_sc;
    if (fftSize == 64) {
        int null[12] = { 0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63 };
        null_sc.assign(null, null + 12);
    }
    return null_sc;
}

std::vector<std::vector<int>> CommsLib::getPilotSc(int fftSize)
{
    std::vector<std::vector<int>> pilot_sc;
    pilot_sc.resize(2);
    if (fftSize == 64) {
        int sc_ind[4] = { 7, 21, 43, 57 };
        int sc_val[4] = { 1, 1, -1, 1 };
        pilot_sc[0].assign(sc_ind, sc_ind + 4);
        pilot_sc[1].assign(sc_val, sc_val + 4);
    }
    return pilot_sc;
}

std::vector<std::complex<float>> CommsLib::IFFT(std::vector<std::complex<float>> in, int fftSize)
{
    std::vector<std::complex<float>> out(in.size());

    void* fft_in = mufft_alloc(fftSize * sizeof(std::complex<float>));
    void* fft_out = mufft_alloc(fftSize * sizeof(std::complex<float>));
    mufft_plan_1d* mufftplan = mufft_create_plan_1d_c2c(fftSize, MUFFT_INVERSE, MUFFT_FLAG_CPU_ANY);

    memcpy(fft_in, in.data(), fftSize * sizeof(std::complex<float>));
    mufft_execute_plan_1d(mufftplan, fft_out, fft_in);
    memcpy(out.data(), fft_out, fftSize * sizeof(std::complex<float>));
    //for (int i = 0; i < fftsize; i++) out[i] /= fftsize;
    float max_val = 0;
    //int max_ind = 0;
    float scale = 0.5;
    for (int i = 0; i < fftSize; i++) {
        if (std::abs(out[i]) > max_val) {
            max_val = std::abs(out[i]);
            //max_ind = i;
        }
    }
    std::cout << "IFFT output is normalized with " << std::to_string(max_val) << std::endl;
    //std::cout << "max sample is " << std::to_string(out[max_ind].real()) << "+1j*" << std::to_string(out[max_ind].imag()) << std::endl;
    for (int i = 0; i < fftSize; i++)
        out[i] /= (max_val / scale);

    mufft_free_plan_1d(mufftplan);
    mufft_free(fft_in);
    mufft_free(fft_out);
    return out;
}

std::vector<std::complex<float>> CommsLib::FFT(std::vector<std::complex<float>> in, int fftSize)
{
    std::vector<std::complex<float>> out(in.size());

    void* fft_in = mufft_alloc(fftSize * sizeof(std::complex<float>));
    void* fft_out = mufft_alloc(fftSize * sizeof(std::complex<float>));
    mufft_plan_1d* mufftplan = mufft_create_plan_1d_c2c(fftSize, MUFFT_FORWARD, MUFFT_FLAG_CPU_ANY);

    memcpy(fft_in, in.data(), fftSize * sizeof(std::complex<float>));
    mufft_execute_plan_1d(mufftplan, fft_out, fft_in);
    memcpy(out.data(), fft_out, fftSize * sizeof(std::complex<float>));

    mufft_free_plan_1d(mufftplan);
    mufft_free(fft_in);
    mufft_free(fft_out);
    return out;
}

std::vector<std::complex<float>> CommsLib::modulate(std::vector<int> in, int type)
{
    std::vector<std::complex<float>> out(in.size());
    if (type == QPSK) {
        float qpsk_table[2][4]; // = init_qpsk();
        float scale = 1 / sqrt(2);
        float mod_qpsk[2] = { -scale, scale };
        for (int i = 0; i < 4; i++) {
            qpsk_table[0][i] = mod_qpsk[i / 2];
            qpsk_table[1][i] = mod_qpsk[i % 2];
        }
        for (size_t i = 0; i < in.size(); i++) {
            if (in[i] >= 0 and in[i] < 4)
                out[i] = std::complex<float>(qpsk_table[0][in[i]], qpsk_table[1][in[i]]);
            else {
                std::cout << "Error: No compatible input vector!" << std::endl;
                break;
            }
        }
    } else if (type == QAM16) {
        float qam16_table[2][16]; //= init_qam16();
        float scale = 1 / sqrt(10);
        float mod_16qam[4] = { -3 * scale, -1 * scale, 3 * scale, scale };
        for (int i = 0; i < 16; i++) {
            qam16_table[0][i] = mod_16qam[i / 4];
            qam16_table[1][i] = mod_16qam[i % 4];
        }
        for (size_t i = 0; i < in.size(); i++) {
            if (in[i] >= 0 and in[i] < 16)
                out[i] = std::complex<float>(qam16_table[0][in[i]], qam16_table[1][in[i]]);
            else {
                std::cout << "Error: No compatible input vector!" << std::endl;
                break;
            }
        }
    } else if (type == QAM64) {
        float qam64_table[2][64]; // = init_qam64();
        float scale = 1 / sqrt(42);
        float mod_64qam[8] = { -7 * scale, -5 * scale, -3 * scale, -1 * scale, scale, 3 * scale, 5 * scale, 7 * scale };
        for (int i = 0; i < 64; i++) {
            qam64_table[0][i] = mod_64qam[i / 8];
            qam64_table[1][i] = mod_64qam[i % 8];
        }
        for (size_t i = 0; i < in.size(); i++) {
            if (in[i] >= 0 and in[i] < 64)
                out[i] = std::complex<float>(qam64_table[0][in[i]], qam64_table[1][in[i]]);
            else {
                std::cout << "Error: No compatible input vector!" << std::endl;
                break;
            }
        }
    } else {
        // Not Supported
        std::cout << "Modulation Type " << type << " not supported!" << std::endl;
    }
    return out;
}

std::vector<std::vector<double>> CommsLib::getSequence(int N, int type)
{
    std::vector<std::vector<double>> matrix;

    if (type == STS_SEQ) {
        // STS - 802.11 Short training sequence (one symbol)
        matrix.resize(2);

        // Normalized sequences
        double sts_re[16] = {
            0.29137692, -0.9219788, -0.04527321, 1.0,
            0.63942015, 1.0, -0.04527321, -0.9219788,
            0.29137692, 0.09784041, -0.53748065, -0.17586154,
            -0.05666628, -0.17586154, -0.53748065, 0.09784041
        };
        double sts_im[16] = {
            0.34804323, -0.06531818, -0.5540778, 0.0,
            0.05666628, 0.0, -0.5540778, -0.06531818,
            0.34804323, -0.91909665, -0.1420086, 0.9844149,
            0.63942015, 0.9844149, -0.1420086, -0.91909665
        };

        int size = sizeof(sts_re) / sizeof(sts_re[0]);
        matrix[0].resize(size);
        matrix[1].resize(size);
        for (int i = 0; i < size; i++) {
            matrix[0][i] = sts_re[i];
            matrix[1][i] = sts_im[i];
        }
    } else if (type == LTS_SEQ) {
        // LTS - 802.11 Long training sequence (160 samples == 2.5 symbols, cp length of 32 samples)
        matrix.resize(2);

        double lts_re[160] = { -0.15625, 0.012284590458567165, 0.09171654912240956, -0.09188755526278, -0.002805944173488664,
            0.07507369706822604, -0.12732435990770957, -0.12188700906074086, -0.03504126073623884, -0.056455128448539,
            -0.060310100316213804, 0.06955684740689412, 0.08221832230305733, -0.1312626089753594, -0.05720634587149917,
            0.03691794200106715, 0.0625, 0.1192390885103326, -0.022483206307774027, 0.05866876712873733,
            0.0244758515211019, -0.13680487681585982, 0.0009889797089880949, 0.05333773437415131, 0.09754126073623881,
            -0.03831596747441851, -0.11513121478170157, 0.05982384485901423, 0.021111770349329442, 0.09683188459112747,
            0.0397496983535005, -0.005121250360419827, 0.15625, -0.005121250360419823, 0.0397496983535005,
            0.09683188459112749, 0.02111177034932945, 0.05982384485901426, -0.11513121478170157, -0.0383159674744185,
            0.09754126073623884, 0.05333773437415131, 0.0009889797089880983, -0.1368048768158598, 0.024475851521101908,
            0.05866876712873735, -0.02248320630777403, 0.1192390885103326, 0.0625, 0.03691794200106713,
            -0.05720634587149916, -0.1312626089753594, 0.08221832230305731, 0.06955684740689413, -0.0603101003162138,
            -0.05645512844853901, -0.03504126073623881, -0.12188700906074088, -0.12732435990770957, 0.07507369706822604,
            -0.002805944173488671, -0.09188755526278002, 0.09171654912240956, 0.01228459045856714, -0.15625,
            0.012284590458567165, 0.09171654912240956, -0.09188755526278, -0.002805944173488664, 0.07507369706822604,
            -0.12732435990770957, -0.12188700906074086, -0.03504126073623884, -0.056455128448539, -0.060310100316213804,
            0.06955684740689412, 0.08221832230305733, -0.1312626089753594, -0.05720634587149917, 0.03691794200106715,
            0.0625, 0.1192390885103326, -0.022483206307774027, 0.05866876712873733, 0.0244758515211019,
            -0.13680487681585982, 0.0009889797089880949, 0.05333773437415131, 0.09754126073623881, -0.03831596747441851,
            -0.11513121478170157, 0.05982384485901423, 0.021111770349329442, 0.09683188459112747, 0.0397496983535005,
            -0.005121250360419827, 0.15625, -0.005121250360419823, 0.0397496983535005, 0.09683188459112749,
            0.02111177034932945, 0.05982384485901426, -0.11513121478170157, -0.0383159674744185, 0.09754126073623884,
            0.05333773437415131, 0.0009889797089880983, -0.1368048768158598, 0.024475851521101908, 0.05866876712873735,
            -0.02248320630777403, 0.1192390885103326, 0.0625, 0.03691794200106713, -0.05720634587149916,
            -0.1312626089753594, 0.08221832230305731, 0.06955684740689413, -0.0603101003162138, -0.05645512844853901,
            -0.03504126073623881, -0.12188700906074088, -0.12732435990770957, 0.07507369706822604, -0.002805944173488671,
            -0.09188755526278002, 0.09171654912240956, 0.01228459045856714, -0.15625, 0.012284590458567165,
            0.09171654912240956, -0.09188755526278, -0.002805944173488664, 0.07507369706822604, -0.12732435990770957,
            -0.12188700906074086, -0.03504126073623884, -0.056455128448539, -0.060310100316213804, 0.06955684740689412,
            0.08221832230305733, -0.1312626089753594, -0.05720634587149917, 0.03691794200106715, 0.0625,
            0.1192390885103326, -0.022483206307774027, 0.05866876712873733, 0.0244758515211019, -0.13680487681585982,
            0.0009889797089880949, 0.05333773437415131, 0.09754126073623881, -0.03831596747441851, -0.11513121478170157,
            0.05982384485901423, 0.021111770349329442, 0.09683188459112747, 0.0397496983535005, -0.005121250360419827 };

        double lts_im[160] = { 0.0, -0.09759955359207202, -0.10587165981863113, -0.11512870891096853, -0.053774266476545984,
            0.07404041892509948, 0.020501379986300285, 0.01656621813913718, 0.15088834764831843, 0.021803920607437133,
            -0.08128612411572139, -0.014121958590578302, -0.09235655195372787, -0.06522722901814465, -0.039298588174111096,
            -0.0983441502870872, 0.0625, 0.004095594414801514, -0.1606573329526341, 0.01493899945069943,
            0.05853179569459056, 0.04737981136568012, 0.11500464362403023, -0.0040763264805083466, 0.025888347648318433,
            0.10617091261510256, 0.05518049537437035, 0.08770675983572167, -0.027885918828227545, -0.08279790948776067,
            0.11115794305116433, 0.12032513267372755, 0.0, -0.1203251326737275, -0.11115794305116432,
            0.08279790948776065, 0.027885918828227538, -0.0877067598357217, -0.05518049537437036, -0.10617091261510254,
            -0.025888347648318433, 0.00407632648050834, -0.11500464362403023, -0.04737981136568013, -0.05853179569459056,
            -0.014938999450699438, 0.16065733295263412, -0.0040955944148015275, -0.0625, 0.09834415028708718,
            0.0392985881741111, 0.06522722901814465, 0.09235655195372787, 0.014121958590578316, 0.08128612411572139,
            -0.021803920607437126, -0.15088834764831843, -0.01656621813913719, -0.02050137998630029, -0.07404041892509945,
            0.05377426647654598, 0.11512870891096855, 0.10587165981863114, 0.09759955359207204, 0.0,
            -0.09759955359207202, -0.10587165981863113, -0.11512870891096853, -0.053774266476545984, 0.07404041892509948,
            0.020501379986300285, 0.01656621813913718, 0.15088834764831843, 0.021803920607437133, -0.08128612411572139,
            -0.014121958590578302, -0.09235655195372787, -0.06522722901814465, -0.039298588174111096, -0.0983441502870872,
            0.0625, 0.004095594414801514, -0.1606573329526341, 0.01493899945069943, 0.05853179569459056,
            0.04737981136568012, 0.11500464362403023, -0.0040763264805083466, 0.025888347648318433, 0.10617091261510256,
            0.05518049537437035, 0.08770675983572167, -0.027885918828227545, -0.08279790948776067, 0.11115794305116433,
            0.12032513267372755, 0.0, -0.1203251326737275, -0.11115794305116432, 0.08279790948776065,
            0.027885918828227538, -0.0877067598357217, -0.05518049537437036, -0.10617091261510254, -0.025888347648318433,
            0.00407632648050834, -0.11500464362403023, -0.04737981136568013, -0.05853179569459056, -0.014938999450699438,
            0.16065733295263412, -0.0040955944148015275, -0.0625, 0.09834415028708718, 0.0392985881741111,
            0.06522722901814465, 0.09235655195372787, 0.014121958590578316, 0.08128612411572139, -0.021803920607437126,
            -0.15088834764831843, -0.01656621813913719, -0.02050137998630029, -0.07404041892509945, 0.05377426647654598,
            0.11512870891096855, 0.10587165981863114, 0.09759955359207204, 0.0, -0.09759955359207202,
            -0.10587165981863113, -0.11512870891096853, -0.053774266476545984, 0.07404041892509948, 0.020501379986300285,
            0.01656621813913718, 0.15088834764831843, 0.021803920607437133, -0.08128612411572139, -0.014121958590578302,
            -0.09235655195372787, -0.06522722901814465, -0.039298588174111096, -0.0983441502870872, 0.0625,
            0.004095594414801514, -0.1606573329526341, 0.01493899945069943, 0.05853179569459056, 0.04737981136568012,
            0.11500464362403023, -0.0040763264805083466, 0.025888347648318433, 0.10617091261510256, 0.05518049537437035,
            0.08770675983572167, -0.027885918828227545, -0.08279790948776067, 0.11115794305116433, 0.12032513267372755 };

        // Grab the last N samples (sequence length specified, provide more flexibility)
        int size = sizeof(lts_re) / sizeof(lts_re[0]);
        int startIdx = size - N;
        matrix[0].resize(N);
        matrix[1].resize(N);
        for (int i = 0; i < N; i++) {
            matrix[0][i] = lts_re[i + startIdx];
            matrix[1][i] = lts_im[i + startIdx];
        }
    } else if (type == LTE_ZADOFF_CHU) {
        // LTE Zadoff Chu Sequence: Generate the 25th root length-63 Zadoff-Chu sequence
        matrix.resize(2);

        double lts_re[63] = { 1.0, -0.7971325072229225, 0.3653410243663958, -0.7330518718298251, 0.9801724878485435,
            0.955572805786141, -0.49999999999999617, 0.7660444431189757, -0.222520933956311, 0.6234898018587135,
            0.4562106573531701, 0.3653410243663966, 0.9555728057861371, 0.7660444431189751, -0.49999999999995753,
            -0.7330518718298601, 0.9801724878485425, -0.22252093395630812, 0.6234898018586816, -0.7971325072229237,
            -0.5000000000000849, -0.5000000000000051, -0.7971325072228729, -0.9888308262251311, 0.9555728057861521,
            0.9801724878485374, -0.22252093395631578, 1.0, 0.7660444431189537, -0.7330518718300307,
            -0.9888308262251518, 0.4562106573531763, -0.9888308262251305, -0.733051871829836, 0.76604444311897,
            1.0, -0.22252093395601577, 0.9801724878485049, 0.9555728057861584, -0.988830826225147,
            -0.797132507222964, -0.4999999999997504, -0.4999999999996758, -0.7971325072227249, 0.6234898018583583,
            -0.2225209339562393, 0.9801724878485397, -0.7330518718300426, -0.5000000000003123, 0.7660444431190734,
            0.9555728057861007, 0.3653410243666958, 0.4562106573529356, 0.6234898018587859, -0.22252093395649927,
            0.7660444431189057, -0.5000000000002512, 0.9555728057860857, 0.9801724878483727, -0.7330518718292615,
            0.3653410243664768, -0.797132507222648, 1.0 };

        double lts_im[63] = { 0.0, -0.6038044103254774, -0.9308737486442039, -0.6801727377709207, 0.1981461431993993,
            0.2947551744109033, -0.8660254037844408, -0.642787609686542, -0.9749279121818244, 0.7818314824680458,
            0.8898718088114649, -0.9308737486442037, 0.294755174410916, -0.6427876096865428, 0.8660254037844631,
            0.680172737770883, 0.19814614319940435, 0.9749279121818251, 0.7818314824680712, -0.6038044103254757,
            -0.8660254037843896, 0.8660254037844357, -0.6038044103255429, 0.1490422661761573, -0.2947551744108673,
            0.19814614319942933, -0.9749279121818233, -6.273657903199343e-14, -0.6427876096865683, 0.6801727377706992,
            0.14904226617601968, 0.8898718088114618, 0.14904226617616118, 0.6801727377709089, -0.6427876096865488,
            -6.666535247945037e-14, -0.9749279121818918, 0.1981461431995907, -0.2947551744108467, 0.14904226617605265,
            -0.6038044103254225, 0.8660254037845827, -0.8660254037846258, -0.6038044103257382, 0.781831482468329,
            0.9749279121818407, 0.19814614319941776, 0.6801727377706862, 0.8660254037842583, -0.6427876096864258,
            0.29475517441103394, -0.9308737486440862, 0.8898718088115852, 0.7818314824679881, -0.9749279121817814,
            -0.6427876096866254, -0.8660254037842936, 0.2947551744110826, 0.19814614320024387, -0.6801727377715282,
            -0.9308737486441722, -0.6038044103258398, 1.021155254707157e-12 };

        int size = sizeof(lts_re) / sizeof(lts_re[0]);
        matrix[0].resize(size);
        matrix[1].resize(size);
        for (int i = 0; i < size; i++) {
            matrix[0][i] = lts_re[i];
            matrix[1][i] = lts_im[i];
        }
    } else if (type == GOLD_IFFT) {
        // Gold IFFT Sequence - seq_length=128, cp=0, upsample=1
        matrix.resize(2);

        double lts_re[128] = { -0.5646359, 0.4669951, 0.8769358, 0.5407985, -0.48144832,
            -0.88476783, 0.33639774, -0.43609348, -0.26278743, 0.6910331,
            -0.25535262, 0.11774132, 0.46892625, 0.77644444, -0.14834122,
            -0.13464923, -0.26617187, 0.1341292, 0.133574, 0.15594807,
            -0.057847068, 0.3967621, 0.047606125, 0.01414329, 0.41560003,
            0.12632199, -0.33603117, -0.5669182, -0.2004348, 0.55602646,
            0.24340886, -0.16611233, 0.7904902, -0.42025912, -0.38651145,
            -0.14808364, -0.27662534, -0.74715126, 0.5908927, -0.75451213,
            -0.33933204, 0.36646086, -0.57852495, 0.10015667, -0.34719938,
            0.35134, 0.7383081, -0.3743101, -0.53234375, -0.33714586,
            0.012157675, -0.399321, -0.3871609, 0.27705255, 0.4469853,
            -0.16857521, 0.60894567, -0.04652265, 0.21421923, 0.014229958,
            0.87569416, -0.28046992, 0.64841086, 0.06317055, -0.037642393,
            -0.7303067, 0.6826409, -0.091142215, -0.080362685, 0.1991867,
            0.3268059, 0.6429179, 0.26278743, -0.088880904, 0.25250778,
            0.2633651, -0.7295981, -0.15740044, -0.44250035, -0.0022179564,
            0.26617187, -0.33556038, -0.38437498, -0.8211783, 0.641319,
            0.3527957, -0.062620886, 0.4227164, -0.23919682, 0.18401834,
            -0.14366682, 0.016121548, -0.25830117, 0.82918876, 0.92221844,
            0.31633607, -0.18821196, -0.9082796, 0.11038142, 0.008659021,
            -0.18971694, -0.40438867, -0.12019706, -0.6811534, 0.33933204,
            -0.40837204, 0.22615194, 0.38991654, 0.18199626, -0.1321399,
            0.19951832, 0.7384663, 0.53234375, 0.030798966, 0.40922493,
            0.4283689, -0.37271422, 0.22344504, 0.24096492, 0.1736422,
            0.4192076, -0.42793053, 0.37122476, -0.008662291, 0.008916863,
            0.34757638, -0.35418823, 0.3462311 };

        double lts_im[128] = { -0.5646359, 0.3462311, -0.35418823, 0.34757638, 0.008916863,
            -0.008662291, 0.37122476, -0.42793053, 0.4192076, 0.1736422,
            0.24096492, 0.22344504, -0.37271422, 0.4283689, 0.40922493,
            0.030798966, 0.53234375, 0.7384663, 0.19951832, -0.1321399,
            0.18199626, 0.38991654, 0.22615194, -0.40837204, 0.33933204,
            -0.6811534, -0.12019706, -0.40438867, -0.18971694, 0.008659021,
            0.11038142, -0.9082796, -0.18821196, 0.31633607, 0.92221844,
            0.82918876, -0.25830117, 0.016121548, -0.14366682, 0.18401834,
            -0.23919682, 0.4227164, -0.062620886, 0.3527957, 0.641319,
            -0.8211783, -0.38437498, -0.33556038, 0.26617187, -0.0022179564,
            -0.44250035, -0.15740044, -0.7295981, 0.2633651, 0.25250778,
            -0.088880904, 0.26278743, 0.6429179, 0.3268059, 0.1991867,
            -0.080362685, -0.091142215, 0.6826409, -0.7303067, -0.037642393,
            0.06317055, 0.64841086, -0.28046992, 0.87569416, 0.014229958,
            0.21421923, -0.04652265, 0.60894567, -0.16857521, 0.4469853,
            0.27705255, -0.3871609, -0.399321, 0.012157675, -0.33714586,
            -0.53234375, -0.3743101, 0.7383081, 0.35134, -0.34719938,
            0.10015667, -0.57852495, 0.36646086, -0.33933204, -0.75451213,
            0.5908927, -0.74715126, -0.27662534, -0.14808364, -0.38651145,
            -0.42025912, 0.7904902, -0.16611233, 0.24340886, 0.55602646,
            -0.2004348, -0.5669182, -0.33603117, 0.12632199, 0.41560003,
            0.01414329, 0.047606125, 0.3967621, -0.057847068, 0.15594807,
            0.133574, 0.1341292, -0.26617187, -0.13464923, -0.14834122,
            0.77644444, 0.46892625, 0.11774132, -0.25535262, 0.6910331,
            -0.26278743, -0.43609348, 0.33639774, -0.88476783, -0.48144832,
            0.5407985, 0.8769358, 0.4669951 };

        int size = sizeof(lts_re) / sizeof(lts_re[0]);
        matrix[0].resize(size);
        matrix[1].resize(size);
        for (int i = 0; i < size; i++) {
            matrix[0][i] = lts_re[i];
            matrix[1][i] = lts_im[i];
        }
    } else if (type == HADAMARD) {
        // Hadamard - using Sylvester's construction for powers of 2.
        matrix.resize(N);
        if ((N & (N - 1)) == 0) {
            for (int i = 0; i < N; i++) {
                matrix[i].resize(N);
                for (int j = 0; j < N; j++)
                    matrix[i][j] = hadamard2(i, j);
            }
        }
    }
#if DEBUG_PRINT
    std::cout << "Num elements in first vector: \t " << matrix[0].size()
              << "   Number of rows: " << matrix.size() << std::endl;
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[i].size(); j++) {
            std::cout << "Values[" << i << "][" << j << "]: \t " << matrix[i][j] << std::endl;
        }
    }
#endif
    return matrix;
}

/*
int main(int argc, char *argv[])
{
    std::vector<std::vector<double> > sequence;
    int type = CommsLib::LTS_SEQ; //atoi(argv[1]);
    int N = 160; 			  //atoi(argv[2]); 	// If Hadamard, possible N: {2, 4, 8, 16, 32, 64}
    sequence = CommsLib::getSequence(N, type);

    std::vector<std::complex<double>> sequence_c;
    for(int i=0; i<sequence[0].size(); i++){
        sequence_c.push_back(std::complex<double>(sequence[0][i], sequence[1][i]));
    }
    double peak = CommsLib::findLTS(sequence_c, N);
    std::cout << "LTS PEAK: " << peak << std::endl;

    return 0;
}
*/
