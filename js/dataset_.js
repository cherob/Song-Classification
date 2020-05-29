const fs = require(`fs`);
const wf = require('wavefile');
const cliProgress = require('cli-progress');
const MFCC = require('./mfcc/mfcc.js');
const S = require('./util.js').shape;
const Config = require('./cfg.v2').Config;
const Mp32Wav = require('mp3-to-wav');

class Dataset {

	/**
	 * @param {Config} config Config file
	 */
	constructor(config) {
		this.config = config;
		this.shape = undefined;
		this.unclean = false;
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
		let path = [
			__dirname,
			'..',
			this.config.audio_path,
			genre,
			file
		];
		let typ = file.match(/\.[a-zA-Z0-9]+/gm);
		typ = typ[typ.length - 1];
		let file_localpath = path.join('\\');

		if (typ == '.mp3') {
			let otherfile = file_localpath;
			otherfile = otherfile.replace(/\.[a-zA-Z0-9]+/gm, '.wav');
			if (fs.existsSync(otherfile)) {
				console.log('removing other File...')
				fs.unlinkSync(file_localpath);
			}
			if (!this.unclean) {
				console.warn('Unclean Data...');
			}
			this.unclean = true;
			return;
		} else if (typ != '.wav') {
			throw new Error('Not supported file: ' + file_localpath);
		}

		let buffer = fs.readFileSync(file_localpath);
		let wav = new wf.WaveFile(buffer);

		if (wav.fmt.sampleRate !== this.config.frame_rate) {
			wav.toSampleRate(this.config.frame_rate);
			fs.writeFileSync(file_localpath, wav.toBuffer());
		}

		wav = wav.toBuffer();
		return wav;
	}


	getSamples(isForTesting = false) {
		this.mfcc_conf = {
			cep_lifter: this.config.cep_lifter,
			frame_rate: this.config.frame_rate,
			fsize: this.config.fsize,
			fstep: this.config.fstep,
			nfft: this.config.nfft,
			nfilt: this.config.nfilt,
			nstep: this.config.nstep,
			num_ceps: this.config.num_ceps,
			pre_emphasis: this.config.pre_emphasis
		}

		let X = [];
		let y = [];

		// create a new progress bar instance and use shades_classic theme
		this.bar1 = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
		this.bar1.start(isForTesting ? this.config.validation_len : this.config.files_len, 0);
		this.progress = 0;
		for (let i = 0; i < this.config.files_per_class; i++) {
			if (isForTesting && this.config.validation_len <= X.length) break;
			Object.keys(this.config.files).forEach(genre => {
				let wavBufferOrg = this.getFileBuffer(genre, i, this.config);
				if (this.unclean) return;
				let wavBuffer = wavBufferOrg.slice(this.config.frame_rate * 20, wavBufferOrg.length - (this.config.frame_rate * 10))

				if (wavBuffer.length < this.config.step)
					wavBuffer = wavBufferOrg.slice(this.config.frame_rate * 10, wavBufferOrg.length)

				if (isForTesting) {
					let MFCCs = new MFCC(this.mfcc_conf)
					let random_offset = Math.floor(Math.random() * (wavBuffer.length - this.config.step))
					let signal = wavBuffer.slice(random_offset, random_offset + this.config.step)

					// load cepstral parts
					let mfccs = [];
					mfccs = MFCCs.getMFCCs(signal);

					X.push(mfccs);
					y.push(Object.keys(this.config.files).indexOf(genre));

					// progress loading bar
					this.progress++;
					this.bar1.update(this.progress);
				} else {
					let offset = Math.round(wavBuffer.length / (this.config.samples_per_file + 1));

					for (let ii = 0; ii < this.config.samples_per_file; ii++) {
						let MFCCs = new MFCC(this.mfcc_conf)

						let local_offset = offset * ii;
						let signal = wavBuffer.slice(local_offset, local_offset + this.config.step);
						// load cepstral parts
						let mfccs = [];
						mfccs = MFCCs.getMFCCs(signal);

						X.push(mfccs);
						y.push(Object.keys(this.config.files).indexOf(genre));

						// progress loading bar
						this.progress++;
						this.bar1.update(this.progress);
					}
				}
			});
		}
		this.bar1.stop();


		return {
			X,
			y,
		};
	}

	build() {
		if (!this.config.use_checkpoints || this.shape == undefined) {
			// TRAINING DATA
			console.log(` -> building training samples`);
			this.data.train = this.getSamples();
		}

		// TESTING DATA
		console.log(` -> building testing samples`);
		this.data.test = this.getSamples(true);

		this.save()
		this.init();
	}

	init() {
		Object.keys(this.data).forEach(key => {
			this.data[key].X = this.data[key].X.map(c1 => c1.map(c2 => c2.map(c3 => formatValue(c3, this.config.capsuled))));
		});

		this.shape = S(this.data.train.X);
		this.shape.shift()

		console.log(` -> shape: [${S(this.data.train.X)}]`);
	}

	load() {
		console.log(" -> try to load test/train data...")

		if (!fs.existsSync(this.config.data_path)) {
			console.warn(` -> no test/train data (${this.config.data_path}) found...`);
			return
		}

		let jsonData;
		let stringData = fs.readFileSync(this.config.data_path);

		try {
			jsonData = JSON.parse(stringData);
		} catch (error) {
			console.warn(` -> no vailable test/train data (${this.config.data_path}) found...`);
			return;
		}

		if (jsonData.train.X.length <= 1) {
			console.warn(` -> no vailable test/train data (${this.config.data_path}) found...`);
			return;
		}

		this.data = jsonData;
		console.log(` -> loaded data (${this.data.train.X.length} samples)`);
		this.init();
		return true;
	}

	save() {
		let jsonData = this.data;
		let stringData = JSON.stringify(jsonData);

		console.log(` -> save test/train (${this.config.data_path}) data...`)
		fs.writeFileSync(this.config.data_path, stringData);
	}
}


function formatValue(value = Array, innerValue = false) {
	let c1 = value;

	if (value && value[0] && value[0][0])
		c1 = innerValue ? value[0] : value[0][0];
	else if (value[0])
		c1 = innerValue ? value : value[0];
	else
		c1 = innerValue ? [value] : value;

	return c1;
}

function getArrayDepth(value) {
	return Array.isArray(value) ?
		1 + Math.max(...value.map(getArrayDepth)) :
		0;
}

module.exports = {
	Dataset: Dataset,
};