const fs = require(`fs`);
const WaveFile = require('wavefile').WaveFile;
const cliProgress = require('cli-progress');
const MFCC = require('mfcc');
const nj = require('numjs');
const join = require('path').join;
const util = require('./util/util.js');
const Config = require('./cfg').Config;

let msg_list = [];
class Dataset {

	/**
	 * @param {Config} config Config file
	 */
	constructor(config) {
		this.config = config;
		this.shape = undefined;
		this.data = {
			train: {
				X: [],
				y: [],
			},
			test: {
				X: [],
				y: [],
			},
		};
	}

	setDataRange(mfcc_data) {
		let _max, _min;

		_max = Math.max(...util.flatten(mfcc_data));
		_min = Math.min(...util.flatten(mfcc_data));

		this.config.max = _max > this.config.max ? _max : this.config.max;
		this.config.min = _min < this.config.min ? _min : this.config.min;
	}

	getFileBuffer(genre, index) {
		let file = this.config.files[genre][index];
		let file_localpath = join(
			__dirname,
			'..',
			this.config.audio_path,
			genre,
			file
		);
		let buffer = fs.readFileSync(file_localpath);
		return buffer;
	}

	/**
	 * 
	 * @param {Array} signal 
	 */
	getMFCCs(signal) {
		signal = new Array(...signal);
		let mfccs = []
		let offset;

		if (!this.config.nsize)
			offset = Math.round(signal.length / this.config.nfeat + 1);

		// printOnce(` -> cepstral overlap: ${this.config.nfft - offset}`);

		// Construct an MFCC with the characteristics we desire
		for (let i = 0; i < this.config.nfeat; i++) {
			let single_part = new Array(this.config.nfft).fill(0)
			let local_offset = this.config.nfft;

			let mfcc = MFCC.construct(
				this.config.nfft, // lenght of single cepstral
				this.config.nfilt, // number of filters (summary of a small frequency range) per cepstral
				this.config.lowfreq, // low frequency cutoff
				this.config.highfreq, // high frequency cutoff
				this.config.frame_rate, // data per second
			);

			if (this.config.nstep)
				local_offset = this.config.nstep * i;
			else
				local_offset = offset * i;


			let sized_signal = signal.slice(local_offset,
				local_offset + this.config.nsize);

			single_part.splice(0, sized_signal.length);
			single_part.unshift(...sized_signal)

			// console.log(single_part);

			// console.log(` -> ${single_part.length} ${local_offset}, ${this.config.nfft}`)
			// Run our MFCC on the FFT magnitudes

			if (single_part.length == this.config.nfft) {
				let mfcc_data = mfcc(single_part, true);

				// add last 4th-Dimension
				let melCoef = mfcc_data.melCoef.map(s => {
					return [s];

				});

				mfccs.push(melCoef);
			}
		}
		// console.log(mfccs.length, mfccs[0].length);
		return mfccs;
	}

	getFixedSamples(isForTesting) {
		let X = [];
		let y = [];

		// create a new progress bar instance and use shades_classic theme
		const bar1 = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
		bar1.start(isForTesting ? this.config.files_len / this.config.samples_per_file : this.config.files_len, 0);
		let progress = 0;
		for (let i = 0; i < this.config.files_per_class; i++) {
			Object.keys(this.config.files).forEach(genre => {
				let wavBuffer = this.getFileBuffer(genre, i, this.config);
				let wav = new WaveFile(wavBuffer);
				wavBuffer = wav.toBuffer();


				let offset = Math.round(wavBuffer.length / this.config.samples_per_file + 1);
				// printOnce(` -> sample overlap: ${this.config.step - offset}`);
				for (let ii = 0; ii < this.config.samples_per_file; ii++) {
					let local_offset = offset * ii;
					let signal;

					let random_offset = Math.floor(Math.random() * (wavBuffer.length - this.config.step)) + this.config.step  
					if (isForTesting)
						signal = wavBuffer.slice(random_offset, random_offset + this.config.step)
					else
						signal = wavBuffer.slice(local_offset, local_offset + this.config.step)

					// load cepstral parts
					let mfccs = this.getMFCCs(signal);

					// progress loading bar
					progress++;
					bar1.update(progress);

					// transpose
					mfccs = mfccs[0].map((col, iii) => mfccs.map(row => row[iii]));

					X.push(mfccs);
					y.push(Object.keys(this.config.files).indexOf(genre));


					if (!isForTesting)
						this.setDataRange(mfccs); // get the the data range  
					else
						break;
				}
			});
		}
		bar1.stop();

		// set with the same ratios from 0 to 1
		X = X.map(ar => ar.map(arr => arr.map(n => [util.map(n[0], this.config.min, this.config.max, 0, 1)])));

		return {
			X,
			y,
		};
	}

	build() {
		// TRAINING DATA
		if (!this.data.train.X.length) {
			console.log(` -> building training samples`);
			this.data.train = this.getFixedSamples();
			this.save()
		}

		// TESTING DATA
		if (!this.data.test.X.length) {
			console.log(` -> building testing samples`);
			this.data.test = this.getFixedSamples(true);
			this.save()
		}


		this.init();
	}

	init() {
		this.data = {
			train: {
				X: nj.array((new Array(this.data.train.X))[0]),
				y: nj.array((new Array(this.data.train.y))[0])
			},
			test: {
				X: nj.array((new Array(this.data.test.X))[0]),
				y: nj.array((new Array(this.data.test.y))[0])
			}
		}

		console.log(` -> shape: [${this.data.train.X.shape}]`)
		// Reshape the data
		this.data.train.X.reshape(this.data.train.X.shape[0], this.data.train.X.shape[1], this.data.train.X.shape[2], 1);
		this.data.test.X.reshape(this.data.test.X.shape[0], this.data.test.X.shape[1], this.data.test.X.shape[2], 1);

		this.shape = [this.data.train.X.shape[1], this.data.train.X.shape[2], 1];
	}

	load() {
		let jsonData, stringData, compressedData, bufferData;

		try {
			console.log(" -> try to load test/train data...")

			stringData = fs.readFileSync(this.config.data_path);
			jsonData = JSON.parse(stringData);

			if (jsonData.train.X.length > 1) {
				this.data = jsonData;
				console.log(` -> loaded data (${this.data.train.X.length} samples)`);
				this.init();
				return true;
			} else {
				console.warn(` -> no vailable test/train data (${this.config.data_path}) found...`);

			}
		} catch (error) {
			console.warn(` -> no test/train data (${this.config.data_path}) found...`);
		}
	}

	save() {
		let jsonData = this.data;
		let stringData, compressedData, bufferData;

		stringData = JSON.stringify(jsonData);
		console.log(` -> save test/train (${this.config.data_path}) data...`)
		fs.writeFileSync(this.config.data_path, stringData);
	}
}

module.exports = {
	Dataset: Dataset,
};