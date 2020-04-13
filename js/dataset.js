const fs = require(`fs`);
const wf = require('wavefile');
const cliProgress = require('cli-progress');
const MFCC = require('./mfcc/mfcc.js');
const numpy = require('numjs');
const join = require('path').join;
const S = require('./util.js').shape;
const util = require('./util.js');
const Config = require('./cfg').Config;

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

		let wav = new wf.WaveFile(buffer);
		// wav.toSampleRate(this.config.frame_rate);
		if (wav.fmt.sampleRate !== this.config.frame_rate)
			wav.toSampleRate(this.config.frame_rate);

		wav = wav.toBuffer();
		return wav;
	}

	getFixedSamples(isForTesting = false) {
		let X = [];
		let y = [];

		// create a new progress bar instance and use shades_classic theme
		const bar1 = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
		bar1.start(isForTesting ? this.config.validation_len : this.config.files_len, 0);
		let progress = 0;
		for (let i = 0; i < this.config.files_per_class; i++) {
			if (isForTesting && i == 1) break;
			Object.keys(this.config.files).forEach(genre => {
				let wavBuffer = this.getFileBuffer(genre, i, this.config);
				wavBuffer = wavBuffer.slice(this.config.frame_rate * 10, wavBuffer.length)

				let offset = Math.round(wavBuffer.length / (this.config.samples_per_file + 1));

				for (let ii = 0; ii < this.config.samples_per_file; ii++) {
					let MFCCs = new MFCC(this.config.pre_emphasis, this.config.frame_size, this.config.frame_stride, this.config.nfft, this.config.nfilt, this.config.num_ceps, this.config.frame_rate, this.config.cep_lifter)

					let local_offset = offset * ii;
					let mfccs = [];
					let signal;

					let random_offset = Math.floor(Math.random() * (wavBuffer.length - this.config.step)) + this.config.step
					if (isForTesting)
						signal = wavBuffer.slice(random_offset, random_offset + this.config.step)
					else
						signal = wavBuffer.slice(local_offset, local_offset + this.config.step)


					// load cepstral parts
					mfccs = MFCCs.getMFCCs(signal)

					X.push(mfccs);
					y.push(Object.keys(this.config.files).indexOf(genre));

					// progress loading bar
					progress++;
					bar1.update(progress);

					if (isForTesting)
						break;
				}
			});
		}
		bar1.stop();

		let min = 0;
		let max = 0;


		X.forEach(mfccs => mfccs.forEach(mfcc => {
			let mfcc_flat = numpy.flatten(mfcc).tolist();
			mfcc_flat.push(min, max)
			max = Math.max(...mfcc_flat);
			min = Math.min(...mfcc_flat);
		}));
		X = X.map(mfccs => util.compress(mfccs, min, max));

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
		}

		// TESTING DATA
		if (!this.data.test.X.length) {
			console.log(` -> building testing samples`);
			this.data.test = this.getFixedSamples(true);
		}
		
		this.save()
		this.init();
	}

	init() {
		Object.keys(this.data).forEach(key => {
			// add last 4th-Dimension
			this.data[key].X = this.data[key].X.map(c1 => {
				return c1.map(c2 => {
					return c2.map(c3 => {
						if (this.config.mode == 0) {
							if (c3 && c3[0] && c3[0][0])
								return c3[0];
							else if (c3[0])
								return c3;
							else
								return [c3];
						} else if (this.config.mode == 1) {
							if (c3 && c3[0] && c3[0][0])
								return c3[0][0];
							else if (c3[0])
								return c3[0];
							else
								return c3;
						}
					});
				});
			});
		});

		this.shape = S(this.data.train.X);
		this.shape.shift()

		console.log(` -> shape: [${S(this.data.train.X)}]`)
	}

	load() {
		let jsonData, stringData;

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
			console.warn(`    ${error.message}`)
		}
	}

	save() {
		let jsonData = this.data;
		let stringData;

		stringData = JSON.stringify(jsonData);
		console.log(` -> save test/train (${this.config.data_path}) data...`)
		fs.writeFileSync(this.config.data_path, stringData);
	}
}

module.exports = {
	Dataset: Dataset,
};