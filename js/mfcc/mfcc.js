const numpy = require('numjs');
const util = require('./utils-mfcc');
const T = require('./utils-mfcc').T;
const S = require('./utils-mfcc').S;
const mjs = require('mathjs')
const dct = require('dct');
const ft = require('fourier-transform');

class MFCC {
    constructor(pre_emphasis = 0.97, frame_size = 0.025, frame_stride = 0.01, nfft = 512, nfilt = 40, num_ceps = 12, sample_rate = 16000, cep_lifter = 22) {
        this.pre_emphasis = pre_emphasis;
        this.frame_size = frame_size;
        this.frame_stride = frame_stride;
        this.nfft = nfft;
        this.nfilt = nfilt;
        this.num_ceps = num_ceps;
        this.sample_rate = sample_rate;
        this.cep_lifter = cep_lifter;
    }

    /**
     * 
     * @param {Array} signal 
     */
    getMFCCs(signal) {
        this.signal = signal;

        /** Pre-Emphasis
         * he first step is to apply a pre-emphasis filter on the signal to amplify the high frequencies. 
         * A pre-emphasis filter is useful in several ways: 
         *  (1) balance the frequency spectrum since high frequencies usually have smaller magnitudes 
         *      compared to lower frequencies, 
         *  (2) avoid numerical problems during the Fourier transform operation and 
         *  (3) may also improve the Signal-to-Noise Ratio (SNR).
         */

        this.preEmphasis();

        /** Framing
         * After pre-emphasis, we need to split the signal into short-time frames. The rationale behind this 
         * step is that frequencies in a signal change over time, so in most cases it doesn’t make sense to 
         * do the Fourier transform across the entire signal in that we would lose the frequency contours of 
         * the signal over time. To avoid that, we can safely assume that frequencies in a signal are stationary 
         * over a very short period of time. Therefore, by doing a Fourier transform over this short-time frame,
         * we can obtain a good approximation of the frequency contours of the signal by concatenating adjacent frames. 
         * 
         * Typical frame sizes in speech processing range from 20 ms to 40 ms with 50% (+/-10%) overlap between 
         * consecutive frames. Popular settings are 25 ms for the frame size, 
         *  frame_size = 0.025 and a 10 ms stride (15 ms overlap), 
         *  frame_stride = 0.01.
         */

        this.framing();

        /** Window
         * After slicing the signal into frames, we apply a window function such as the Hamming 
         * window to each frame. 
         * 
         * There are several reasons why we need to apply a window function to the frames, notably to 
         * counteract the assumption made by the FFT that the data is infinite and to reduce spectral leakage.
         */
        // this.frames = util.hamming(this.frames, this.nsize);

        /** Fourier-Transform and Power Spectrum
         * We can now do an N-point FFT on each frame to calculate the frequency spectrum, 
         * which is also called Short-Time Fourier-Transform (STFT), where N is typically 256 or 512, 
         * NFFT = 512; and then compute the power spectrum (periodogram).
         */

        this.FFT_PowSpec();

        /** Filter Banks
         * The final step to computing filter banks is applying triangular filters, 
         * typically 40 filters, nfilt = 40 on a Mel-scale to the power spectrum to 
         * extract frequency bands. The Mel-scale aims to mimic the non-linear human ear 
         * perception of sound, by being more discriminative at lower frequencies and less 
         * discriminative at higher frequencies. We can convert between Hertz (f) and Mel (m).
         * 
         * Each filter in the filter bank is triangular having a response of 1 at the center 
         * frequency and decrease linearly towards 0 till it reaches the center frequencies 
         * of the two adjacent filters where the response is 0.
         */

        this.filterBanks();

        /** Mel-frequency Cepstral Coefficients (MFCCs)
         * It turns out that filter bank coefficients computed in the previous step are highly 
         * correlated, which could be problematic in some machine learning algorithms. Therefore, 
         * we can apply Discrete Cosine Transform (DCT) to decorrelate the filter bank coefficients 
         * and yield a compressed representation of the filter banks. Typically, for Automatic 
         * Speech Recognition (ASR), the resulting cepstral coefficients 2-13 are retained and 
         * the rest are discarded; num_ceps = 12. The reasons for discarding the other coefficients 
         * is that they represent fast changes in the filter bank coefficients and these fine 
         * details don’t contribute to Automatic Speech Recognition (ASR).
         */

        this.MFCCs();

        /** Mean Normalization
         * As previously mentioned, to balance the spectrum and improve the Signal-to-Noise (SNR),
         * we can simply subtract the mean of each coefficient from all frames.
         */

        this.meanNormalization();

        /** Value Range Reduction
         * Conversion of all numbers in the range 0.00 to 1.00 
         *  lowest value => 0.00
         *  highest value => 1.00
         */
        this.mfccs = this.mfccs.map(mfcc => mfcc.map(value => -value))



        this.mfccs = T(this.mfccs)
        // this.mfccs = this.mfccs.map((mfcc) => mfcc.reverse())

        return this.return_value || this.mfccs;
    }

    preEmphasis() {
        this.emphasized_signal = new Array(0);
        this.emphasized_signal.push(this.signal[0]);

        let pre_emphasis = this.signal[this.signal.length - 1]
        pre_emphasis = pre_emphasis * this.pre_emphasis;

        let signal_ = this.signal.slice(1, this.signal.length);
        signal_ = signal_.map(value => value - pre_emphasis)

        signal_.forEach(element => {
            this.emphasized_signal.push(element)
        });
    }

    framing() {
        // Convert from seconds to samples
        let frame_length = this.frame_size * this.sample_rate
        let frame_step = this.frame_stride * this.sample_rate

        let signal_length = this.emphasized_signal.length
        frame_length = Math.round(frame_length)
        frame_step = Math.round(frame_step)

        // Make sure that we have at least 1 frame
        let num_frames = mjs.ceil(Math.abs(signal_length - frame_length) / frame_step)

        let pad_signal_length = num_frames * frame_step + frame_length
        let z = mjs.zeros(pad_signal_length - signal_length);
        let pad_signal = this.emphasized_signal.concat(z)
        let cols = numpy.arange(0, num_frames * frame_step, frame_step).tolist();

        this.frames = cols.map((rows) => {
            let frame_length_empty = Math.pow(2, Math.round(Math.log(frame_length) / Math.log(2))) - frame_length;
            rows = pad_signal.slice(rows, rows + frame_length);
            rows = rows.concat(new Array(frame_length_empty).fill(0));
            return rows;
        });
    }

    FFT_PowSpec() {
        // get normalized magnitudes for frequencies from 0 to 22050 with interval 44100/1024 ≈ 43Hz
        this.mag_frames = this.frames.map(frame => {
            let frame_ = new Array(0);
            frame_ = ft(frame)
            return frame_;
        });
        this.mag_frames = this.mag_frames.map(frame => frame.map(value => Math.abs(value)));
        this.pow_frames = this.mag_frames.map(frame => frame.map(value => (1.0 / this.nfft) * Math.pow(value, 2)));
    }

    filterBanks() {
        let low_freq_mel = 0
        let high_freq_mel = (2595 * Math.log10(1 + (this.sample_rate / 2) / 700)) // Convert Hz to Mel
        let mel_points = util.linspace(low_freq_mel, high_freq_mel, this.nfft + 2) // Equally spaced in Mel scale
        let hz_points = mel_points.map(point => (700 * (10 ** (point / 2595) - 1)));
        let bin = hz_points.map(point => Math.floor((this.nfft + 1) * point / this.sample_rate));

        let fbank = mjs.zeros([this.nfilt, mjs.floor(this.pow_frames[0].length)]);
        for (let m = 1; m < (this.nfilt + 1); m++) {
            let f_m_minus = bin[m - 1]
            let f_m = bin[m]
            let f_m_plus = bin[m + 1]

            for (let k = f_m_minus; k < f_m; k++)
                fbank[m - 1][k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])

            for (let k = f_m; k < f_m_plus; k++) {
                fbank[m - 1][k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
            }
        }

        fbank = T(fbank);
        this.filter_banks = numpy.dot(this.pow_frames, fbank).tolist();
        this.filter_banks = this.filter_banks.map(filter_bank => filter_bank.map(filter => 20 * Math.log10(filter)));
        // dB
        this.filter_banks = this.filter_banks.map(filter_bank => filter_bank.map(value => {
            if (value == Infinity) value = 0
            if (value == -Infinity) value = 0
            return value
        }))
    }

    MFCCs() {
        this.mfccs = new Array(0);
        this.filter_banks.forEach(filter_bank => this.mfccs.push(dct(filter_bank).slice(0, this.num_ceps)))

        let ncoeff = S(this.mfccs)[1]

        let n = numpy.arange(ncoeff).tolist()
        let lift = n.map(n_ => 1 + (this.cep_lifter / 2) * Math.sin(Math.PI * n_ / this.cep_lifter));

        this.mfccs = this.mfccs.map(mfcc => {
            mfcc = mfcc.map((value, index) => {
                value *= lift[index];
                return value;
            });
            return mfcc;
        });
    }

    meanNormalization() {
        this.mfccs = this.mfccs.map(mfcc => mfcc = mfcc.map(value => value -= (numpy.mean(mfcc) + 1e-8)));
        this.filter_banks = this.filter_banks.map(filter_bank => filter_bank.map(filter => (numpy.mean(filter) + 1e-8)));
    }


}

module.exports = MFCC;