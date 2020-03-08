const fs = require(`fs`);
const join = require('path').join;

function getID(config) {
	let id = 1;

	let keys = ["nfft", "nfilt", "nfeat",
		"lowfreq", "highfreq", "frame_rate",
		"samples_per_file", "sample_length",
		"files_per_class", "nstep", "nsize"
	];

	keys.forEach(key => {
		let value = config[key];
		if (!value) value = Math.round(Math.PI * 100) / 100;
		let key_length = key.toString().length;
		let value_length = value.toString().length;

		
		id += (value / key_length) * value_length;
	});
	id = `${Math.round(id)}${config.categories}`;
	return id;
}

function getFiles(config) {
	let files = {};

	fs.readdirSync(config.audio_path).forEach(dir => {
		files[dir] = [];
		fs.readdirSync(join(config.audio_path, dir)).forEach((file, index) => {
			if (!config.files_per_class || index < config.files_per_class)
				files[dir].push(file);
		});
	});
	return files;
}

function getFilesPerClass(config) {
	let files_per_class = 0;
	Object.keys(config.files).forEach(dir => {
		if (files_per_class) {
			if (files_per_class > config.files[dir].length) files_per_class = config.files[dir].length;
		} else {
			files_per_class = config.files[dir].length;
		}
	});
	return files_per_class;
}

function getSamplesFromMillis(ms, rate) {
	let frame_per_milli = rate / 1000
	let samples = frame_per_milli * ms;
	return samples;
}

class Config {
	constructor() {
		// Paths
		this.audio_path = join("audio", "refactored");

		// Files
		this.audio_startpoint = 10; // (int)
		this.audio_length = 120; // (int)
		this.validation_data_mult = 1; // (int)

		// Model	
		this.nfft = 512; // lenght of single cepstral
		this.nfilt = 26; // number of filters (summary of a small frequency range) per cepstral
		this.nsize = 25 // (ms) window size of actually cepstal
		this.nstep = 10 // (ms) step size for cepstal overlap
		this.nfeat = 13; // number of amount of cepstral 
		this.lowfreq = 20; // low frequency cutoff
		this.highfreq = 10000; // high frequency cutoff 
		this.frame_rate = 16000; // data per second

		//Files
		this.samples_per_file = 4; // amount of parts to look at (per file)
		this.sample_length = 2; // length of single sample part in second
		this.files_per_class = false // self explaining	

		// Edit
		this.nstep = getSamplesFromMillis(this.nstep, this.frame_rate);
		this.nsize = getSamplesFromMillis(this.nsize, this.frame_rate);
		// this.nfeat = 130 * this.sample_length; // number of amount of cepstral

		// Options
		this.use_checkpoints = true; // (bool)

		// Training
		this.shuffle = false // (bool)
		this.shuffle_fit = true // (bool)
		this.epochs = 5; // (int)
		this.batch_size = 256; // (int)
		this.calls = false; // (bool, int)

		// Global
		this.step = this.frame_rate * this.sample_length; // get amout of data for given sample length

		// Range
		this.min = Number.MAX_VALUE;
		this.max = 0;

		// Dependencies
		this.categories = fs.readdirSync(this.audio_path).length;
		this.id = getID(this);
		this.files = getFiles(this);
		this.files_per_class = getFilesPerClass(this);
		this.files_len = this.categories * this.files_per_class * this.samples_per_file;
		this.files_len = Math.round(this.files_len);

		// this.model_audio_date_path = join(`data`, `model` + this.id + `.csv`);
		// this.predictions_date_path = join(`data`, `predictions` + this.id + `.csv`);
		this.model_path = `file://./models/${this.id}/`;
		this.data_path = join(`data`, this.id + `.json`);
		this.chart_acc_path = join("images", "acc", this.id + `.png`);
		this.chart_loss_path = join("images", "loss", this.id + `.png`);
	}

	calculate() {
		console.log(` ${this.categories} loaded categories`);
		console.log(`   with  ${this.files_per_class} files each,`);
		console.log(`   every file is ${this.audio_length} seconds long,`);
		console.log(`   divided into ${this.samples_per_file} samples`);
		console.log(` -> results in ${this.files_len} data records`);
		console.log(`    ${this.data_path}`);
	}
}

module.exports = {
	Config: Config,
};